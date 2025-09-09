# worker.py
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from copy import deepcopy
from collections import deque
import matplotlib.patches as patches
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec

from env import Env
from agent import Agent
from utils import *
from node_manager import NodeManager
from ground_truth_node_manager import GroundTruthNodeManager
from parameter import *
import math
from rendezvous_picker import pick_rendezvous_point

from attention_viz import dump_attn_debug_pngs, AttnRecorder

if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)

def make_gif_safe(frame_paths, out_path, duration_ms=120):
    frame_paths = [p for p in frame_paths if os.path.exists(p)]
    frame_paths.sort()
    if not frame_paths: return
    frames = []
    base_size = None
    for p in frame_paths:
        try:
            im = Image.open(p).convert("RGB")
            if base_size is None: base_size = im.size
            elif im.size != base_size: im = im.resize(base_size, Image.BILINEAR)
            frames.append(im)
        except Exception: continue
    if not frames: return
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0, optimize=False)

_mission_id_counter = 0
def get_new_mission_id():
    global _mission_id_counter
    _mission_id_counter += 1
    return _mission_id_counter

class Mission:
    # --- FIX: Mission no longer needs ref_map_info ---
    def __init__(self, P, T_meet, r_meet, participants, start_step, meta=None, pending=False):
        self.id = get_new_mission_id()
        self.P = np.array(P, dtype=float)
        self.T_meet = int(T_meet)
        self.r_meet = float(r_meet)
        self.participants = set(participants)
        self.t0 = int(start_step)
        self.meta = meta if isinstance(meta, dict) else {}
        self.pending = pending
        self.color = plt.cm.get_cmap('spring', 10)(self.id % 10)

class Worker:
    def __init__(self, meta_agent_id, policy_net, predictor, global_step, device='cpu', save_image=False, attn_recorder: AttnRecorder=None):
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        self.device = device
        self.env = Env(global_step, plot=self.save_image)
        self.node_manager = NodeManager(plot=self.save_image)
        self.robots = [Agent(i, policy_net, predictor, self.node_manager, device=device, plot=save_image) for i in range(N_AGENTS)]
        self.gtnm = GroundTruthNodeManager(self.node_manager, self.env.ground_truth_info, device=device, plot=save_image)
        self.episode_buffer = [[] for _ in range(27)]
        self.perf_metrics = dict()
        self.last_known_locations = [self.env.robot_locations.copy() for _ in range(N_AGENTS)]
        self.last_known_intents = [{aid: [] for aid in range(N_AGENTS)} for _ in range(N_AGENTS)]
        self.run_dir = os.path.join(gifs_path, f"run_g{self.global_step}_w{self.meta_agent_id}_{os.getpid()}_{int(time.time()*1000)}")
        if self.save_image: os.makedirs(self.run_dir, exist_ok=True)
        self.env.frame_files = []
        self.missions = {}
        self.pending_mission = None
        self.was_fully_connected = False
        self._rdv_paths_per_agent = [set() for _ in range(N_AGENTS)]
        self.disc_free_ts, self.disc_occ_ts = [], []
        self.attn_recorder = attn_recorder
        gt_map = self.env.ground_truth_info.map
        self._gt_free_total = int(np.count_nonzero(gt_map == FREE))


    def _match_intent_channels(self, obs_pack):
        n, m, e, ci, ce, ep = obs_pack
        need, got = NODE_INPUT_DIM, n.shape[-1]
        if got < need:
            pad = torch.zeros((n.shape[0], n.shape[1], need - got), dtype=n.dtype, device=n.device)
            n = torch.cat([n, pad], dim=-1)
        return [n, m, e, ci, ce, ep]

    def run_episode(self):
        """
        单回合运行 + 注意力可视化：
        - 在每个 agent 选择动作的前向推理之后，抓取本次前向产生的注意力，并将
        “当前节点 -> 候选邻居(K_SIZE)”的注意力分布可视化为 PNG。
        - 输出目录：<run_dir>/attn/agent{aid}/t{step}/<layer>_head{h}.png
        """
        done = False

        # -------- 初始化：图、预测、意图 --------
        for i, r in enumerate(self.robots):
            r.update_graph(self.env.get_agent_map(i), self.env.robot_locations[i])
        for r in self.robots:
            r.update_predict_map()
        for i, r in enumerate(self.robots):
            r.update_planning_state(self.env.robot_locations)
            self.last_known_intents[r.id][r.id] = deepcopy(r.intent_seq)

        # 初始连通性与首帧绘图
        groups0 = self.env._compute_comm_groups()
        self.was_fully_connected = (len(groups0) == 1 and len(groups0[0]) == N_AGENTS)
        if self.save_image:
            self.plot_env(step=0)

        # ================= 主循环 =================
        for t in range(MAX_EPISODE_STEP):
            rdv_reward_this_step = 0.0

            # 计算连通性
            groups = self.env._compute_comm_groups()
            is_fully_connected = (len(groups) == 1 and len(groups[0]) == N_AGENTS)

            # --- 统一的全局 MapInfo（给 rendezvous & 路径等用） ---
            a0 = self.robots[0]  # 用 agent0 的预测作为参考
            global_map_info = MapInfo(self.env.global_belief,
                                    self.env.belief_origin_x,
                                    self.env.belief_origin_y,
                                    self.env.cell_size)
            pred_map = (a0.pred_mean_map_info.map
                        if a0.pred_mean_map_info is not None
                        else np.copy(global_map_info.map))
            belief_map = global_map_info.map
            trav_mask = (((pred_map / 255.0) >= 0.6) | (belief_map == FREE)) & (belief_map != OCCUPIED)
            unknown_mask = (belief_map == UNKNOWN)

            # --- 会合任务更新 ---
            if is_fully_connected:
                self.missions.clear()
                try:
                    P, r_meet, T_meet, meta = pick_rendezvous_point(
                        self.robots, t, trav_mask, unknown_mask, global_map_info
                    )
                    if P is not None:
                        self.pending_mission = Mission(P, T_meet, r_meet, range(N_AGENTS), t, meta=meta, pending=True)
                except Exception as e:
                    print(f"[Error] Pending mission update failed: {e}")
            elif self.was_fully_connected and self.pending_mission is not None:
                # 从“待生效”切换到正式任务
                self.pending_mission.pending = False
                self.missions[self.pending_mission.id] = self.pending_mission
                self.pending_mission = None

            if not is_fully_connected:
                missions_to_remove, new_missions_to_add = [], {}
                for mission_id, mission in list(self.missions.items()):
                    # 成功
                    if (not mission.participants) or all(
                        np.linalg.norm(self.robots[aid].location - mission.P) <= mission.r_meet
                        for aid in mission.participants
                    ):
                        rdv_reward_this_step += float(R_MEET_SUCCESS)
                        missions_to_remove.append(mission_id)
                        continue
                    # 超时
                    if t >= mission.T_meet + int(MEET_LATE_TOL):
                        rdv_reward_this_step -= float(R_MEET_LATE)
                        missions_to_remove.append(mission_id)
                        continue

                    # 组内能沟通的参与者拆分出局部任务
                    for group in groups:
                        participants_in_group = mission.participants.intersection(group)
                        if len(participants_in_group) > 1:
                            rep = min(participants_in_group)
                            new_participants = participants_in_group - {rep}
                            mission.participants -= new_participants
                            try:
                                subgroup_robots = [self.robots[i] for i in new_participants]
                                P_new, r_new, T_new, meta_new = pick_rendezvous_point(
                                    subgroup_robots, t, trav_mask, unknown_mask, global_map_info
                                )
                                if P_new is not None:
                                    new_missions_to_add[get_new_mission_id()] = Mission(
                                        P_new, T_new, r_new, new_participants, t, meta=meta_new
                                    )
                            except Exception as e:
                                print(f"[Error] Local mission creation: {e}")

                for mid in missions_to_remove:
                    if mid in self.missions:
                        del self.missions[mid]
                self.missions.update(new_missions_to_add)

            self.was_fully_connected = is_fully_connected

            # 给 agent 写入会合紧迫度，更新 RDV 路径 mask
            agent_mission_info = {
                aid: (m.P, m.T_meet, m.t0) for m in self.missions.values() for aid in m.participants
            }
            for i, ag in enumerate(self.robots):
                if i in agent_mission_info:
                    _, T_meet, t0 = agent_mission_info[i]
                    ag.time_left_norm = float(np.clip(max(0, T_meet - t) / max(1, T_meet - t0), 0.0, 1.0))
                else:
                    ag.time_left_norm = 0.0
            self._update_rdv_paths_per_agent(trav_mask, global_map_info)
            for i, ag in enumerate(self.robots):
                ag.rdv_path_nodes_set = self._rdv_paths_per_agent[i]

            # ---------- 选择动作（含注意力抓取与落图） ----------
            picks, dists = [], []

            for i, r in enumerate(self.robots):
                obs = r.get_observation(self.last_known_locations[i], self.last_known_intents[i])
                c_obs, _ = self.gtnm.get_ground_truth_observation(
                    r.location, r.pred_mean_map_info, self.env.robot_locations
                )
                r.save_observation(obs, self._match_intent_channels(c_obs))

                if self.attn_recorder is not None and self.save_image:
                    self.attn_recorder.begin_step(t)

                nxt, _, act = r.select_next_waypoint(obs)

                if self.attn_recorder is not None and self.save_image:
                    attn_records = self.attn_recorder.end_forward()  
                    try:
                        # obs: [node_inputs, node_padding_mask, edge_mask, current_index_t, current_edge, edge_padding_mask]
                        _, _, _, current_index_t, current_edge, _ = obs
                        current_index = int(current_index_t.item())
                        neighbor_indices = current_edge[0, :, 0].detach().cpu().numpy()
                    except Exception:
                        current_index = 0
                        neighbor_indices = (r.neighbor_indices
                                            if r.neighbor_indices is not None
                                            else np.array([], dtype=int))
                    try:
                        dump_attn_debug_pngs(
                            run_dir=self.run_dir,
                            step=t,
                            agent_id=r.id,
                            records=attn_records,
                            node_coords=r.node_coords,          
                            current_index=current_index,
                            neighbor_indices=neighbor_indices,
                            map_info=r.map_info
                        )
                    except Exception as e:
                        print(f"[AttnViz] Dump failed at t={t}, agent={r.id}: {e}")

                r.save_action(act)
                picks.append(nxt)
                dists.append(np.linalg.norm(nxt - r.location))

            picks = self.resolve_conflicts(picks, dists)
            prev_max = self.env.get_agent_travel().max()
            prev_total = self.env.get_total_travel()
            for r, loc in zip(self.robots, picks):
                self.env.step(loc, r.id)
            self.env.max_travel_dist = self.env.get_agent_travel().max()
            delta_max = self.env.max_travel_dist - prev_max
            delta_total = self.env.get_total_travel() - prev_total

            groups_after_move = self.env._compute_comm_groups()
            for g in groups_after_move:
                for i in g:
                    for j in g:
                        self.last_known_locations[i][j] = self.env.robot_locations[j].copy()
                        self.last_known_intents[i][j] = deepcopy(self.robots[j].intent_seq)

            for i, r in enumerate(self.robots):
                r.update_graph(self.env.get_agent_map(i), self.env.robot_locations[i])
                r.update_predict_map()
                r.update_planning_state(self.last_known_locations[i], self.last_known_intents[i])
                self.last_known_intents[i][r.id] = deepcopy(r.intent_seq)

            '''# ---------- 奖励与终止 ----------
            team_reward_env, per_agent_obs_rewards = self.env.calculate_reward()
            team_reward = (
                team_reward_env
                - ((delta_max / UPDATING_MAP_SIZE) * MAX_TRAVEL_COEF)
                - ((delta_total / UPDATING_MAP_SIZE) * TOTAL_TRAVEL_COEF)
                + rdv_reward_this_step
            )

            utilities_empty = all((r.utility <= 3).all() for r in self.robots)
            done = utilities_empty
            if done:
                team_reward += 10.0'''

            # ---------- 奖励与终止 ----------
            team_reward_env, per_agent_obs_rewards = self.env.calculate_reward()
            team_reward = (
                team_reward_env
                - ((delta_max / UPDATING_MAP_SIZE) * MAX_TRAVEL_COEF)
                - ((delta_total / UPDATING_MAP_SIZE) * TOTAL_TRAVEL_COEF)
                + rdv_reward_this_step
            )

            utilities_empty = all((r.utility <= 3).all() for r in self.robots)

            if self._gt_free_total > 0:
                per_agent_free_counts = [int(np.count_nonzero(r.map_info.map == FREE)) for r in self.robots]
                per_agent_cov = [c / self._gt_free_total for c in per_agent_free_counts]
                coverage_ok = all(c >= 0.995 for c in per_agent_cov)
            else:
                per_agent_cov = [0.0 for _ in self.robots]
                coverage_ok = False

            done = utilities_empty or coverage_ok
            if done:
                team_reward += 10.0

            for i, r in enumerate(self.robots):
                indiv_total = team_reward + per_agent_obs_rewards[i]
                r.save_reward(indiv_total)
                r.save_done(done)
                next_obs = r.get_observation(self.last_known_locations[i], self.last_known_intents[i])
                c_next_obs, _ = self.gtnm.get_ground_truth_observation(
                    r.location, r.pred_mean_map_info, self.env.robot_locations
                )
                r.save_next_observations(next_obs, self._match_intent_channels(c_next_obs))

            if self.save_image:
                self.plot_env(step=t + 1)

            if done:
                break

        self.perf_metrics.update({
            'travel_dist': self.env.get_total_travel(),
            'max_travel': self.env.get_max_travel(),
            'explored_rate': self.env.explored_rate,
            'success_rate': done
        })

        if self.save_image:
            make_gif_safe(self.env.frame_files,
                        os.path.join(self.run_dir, f"ep_{self.global_step}.gif"))

        for r in self.robots:
            for k in range(len(self.episode_buffer)):
                self.episode_buffer[k] += r.episode_buffer[k]

    def resolve_conflicts(self, picks, dists):
        picks = np.array(picks).reshape(-1, 2)
        order = np.argsort(np.array(dists))
        chosen_complex, resolved = set(), [None] * len(self.robots)
        for rid in order:
            robot = self.robots[rid]
            neighbor_coords = sorted(list(robot.node_manager.nodes_dict.find(robot.location.tolist()).data.neighbor_set), key=lambda c: np.linalg.norm(np.array(c) - picks[rid]))
            picked = next((cand for cand in neighbor_coords if complex(cand[0], cand[1]) not in chosen_complex), None)
            resolved[rid] = np.array(picked) if picked else robot.location.copy()
            chosen_complex.add(complex(resolved[rid][0], resolved[rid][1]))
        return np.array(resolved).reshape(-1, 2)

    def _update_rdv_paths_per_agent(self, trav_mask, global_map_info):
        self._rdv_paths_per_agent = [set() for _ in range(N_AGENTS)]
        missions_to_plan_for = list(self.missions.values())
        if self.pending_mission: missions_to_plan_for.append(self.pending_mission)
        if not missions_to_plan_for: return
        
        for mission in missions_to_plan_for:
            goal_rc = self._world_to_cell_rc(mission.P, global_map_info)
            if not trav_mask[goal_rc[0], goal_rc[1]]: goal_rc = self._find_nearest_valid_cell(trav_mask, goal_rc)
            for aid in mission.participants:
                start_rc = self._world_to_cell_rc(self.robots[aid].location, global_map_info)
                if not trav_mask[start_rc[0], start_rc[1]]: start_rc = self._find_nearest_valid_cell(trav_mask, start_rc)
                rc_path = self._bfs_path_rc(trav_mask, start_rc, goal_rc)
                if rc_path: self._rdv_paths_per_agent[aid] |= self._cells_to_nodecoord_set(rc_path, global_map_info)

    def _world_to_cell_rc(self, world_xy, map_info):
        cell = get_cell_position_from_coords(np.array(world_xy), map_info); return int(cell[1]), int(cell[0])
    def _cells_to_nodecoord_set(self, rc_path, map_info):
        if not rc_path: return set()
        coords = np.array([[map_info.map_origin_x + c * map_info.cell_size, map_info.map_origin_y + r * map_info.cell_size] for r, c in rc_path])
        return {tuple(c) for c in np.around(coords, 1)}
    def _find_nearest_valid_cell(self, mask, start_rc):
        q = deque([tuple(start_rc)]); visited = {tuple(start_rc)}
        while q:
            r, c = q.popleft()
            if mask[r, c]: return np.array([r, c])
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and (nr, nc) not in visited:
                    q.append((nr, nc)); visited.add((nr, nc))
        return start_rc
    def _bfs_path_rc(self, trav_mask, start_rc, goal_rc):
        H, W = trav_mask.shape
        q = deque([(tuple(start_rc), [tuple(start_rc)])]); visited = {tuple(start_rc)}
        while q:
            (r, c), path = q.popleft()
            if (r, c) == tuple(goal_rc): return path
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and trav_mask[nr, nc] and (nr, nc) not in visited:
                    visited.add((nr, nc)); q.append(((nr, nc), path + [(nr, nc)]))
        return []

    def plot_env(self, step):
        """
        Global: Belief + (Predicted-free overlay on UNKNOWN)
        Per-agent: [Local Belief (+intents)] | [Local Predicted (+Belief overlay)]
        """
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(18, 10), dpi=110)

        gs = GridSpec(N_AGENTS, 3, figure=fig, width_ratios=[2.5, 1.2, 1.2], wspace=0.15, hspace=0.1)
        ax_global = fig.add_subplot(gs[:, 0])
        ax_locals_obs = [fig.add_subplot(gs[i, 1]) for i in range(N_AGENTS)]
        ax_locals_pred = [fig.add_subplot(gs[i, 2]) for i in range(N_AGENTS)]
        agent_colors = plt.cm.get_cmap('cool', N_AGENTS)

        global_info = MapInfo(self.env.global_belief, self.env.belief_origin_x, self.env.belief_origin_y, self.env.cell_size)
        ax_global.set_title(f"Global View | Step {step}/{MAX_EPISODE_STEP}", fontsize=14, pad=10)
        ax_global.imshow(global_info.map, cmap='gray', origin='lower')
        ax_global.set_aspect('equal', adjustable='box')
        ax_global.set_axis_off()

        if self.robots and self.robots[0].pred_mean_map_info is not None:
            pred_mean = self.robots[0].pred_mean_map_info.map.astype(np.float32) / 255.0  # [0,1]
            belief = global_info.map
            unknown_mask = (belief == UNKNOWN)

            prob = np.zeros_like(pred_mean)
            prob[unknown_mask] = pred_mean[unknown_mask]

            ax_global.imshow(prob, cmap='magma', origin='lower', alpha=0.35) 

        groups = self.env._compute_comm_groups()
        for group in groups:
            for i_idx, i in enumerate(list(group)):
                for j in list(group)[i_idx+1:]:
                    p1 = self._world_to_cell_rc(self.robots[i].location, global_info)
                    p2 = self._world_to_cell_rc(self.robots[j].location, global_info)
                    ax_global.plot([p1[1], p2[1]], [p1[0], p2[0]], color="#33ff88", lw=2, alpha=0.8, zorder=5)

        for i, r in enumerate(self.robots):
            pos_cell = self._world_to_cell_rc(r.location, global_info)
            if hasattr(r, 'trajectory_x') and r.trajectory_x:
                traj_cells = [self._world_to_cell_rc(np.array([x, y]), global_info)
                            for x, y in zip(r.trajectory_x, r.trajectory_y)]
                ax_global.plot([c for r_, c in traj_cells], [r_ for r_, c in traj_cells],
                            color=agent_colors(i), lw=1.5, zorder=3)
            comms_radius = patches.Circle((pos_cell[1], pos_cell[0]), COMMS_RANGE / CELL_SIZE,
                                        fc=(0, 1, 0, 0.05), ec=(0, 1, 0, 0.4), ls='--', lw=1.5, zorder=4)
            ax_global.add_patch(comms_radius)
            ax_global.plot(pos_cell[1], pos_cell[0], 'o', ms=10, mfc=agent_colors(i),
                        mec='white', mew=1.5, zorder=10)

        all_missions_to_plot = list(self.missions.values()) + ([self.pending_mission] if self.pending_mission else [])
        for mission in all_missions_to_plot:
            p_cell = self._world_to_cell_rc(mission.P, global_info)
            if mission.pending:
                ax_global.plot(p_cell[1], p_cell[0], '+', ms=25, c='yellow', mew=3, zorder=12, alpha=0.8)
                radius = patches.Circle((p_cell[1], p_cell[0]), mission.r_meet / CELL_SIZE,
                                        fc=(1, 1, 0, 0.05), ec='yellow', ls=':', lw=2.5, zorder=11)
            else:
                ax_global.plot(p_cell[1], p_cell[0], '*', ms=20, mfc=mission.color, mec='white', mew=2, zorder=12)
                radius = patches.Circle((p_cell[1], p_cell[0]), mission.r_meet / CELL_SIZE,
                                        fc=mission.color, alpha=0.1, ec=mission.color, ls='--', lw=2, zorder=11)
            ax_global.add_patch(radius)
            for aid in mission.participants:
                start_cell = self._world_to_cell_rc(self.robots[aid].location, global_info)
                ax_global.plot([start_cell[1], p_cell[1]], [start_cell[0], p_cell[0]],
                            color=mission.color, ls=':', lw=2, zorder=11)

        for i, r in enumerate(self.robots):
            ax_obs = ax_locals_obs[i]
            local_map_info = r.map_info                
            ax_obs.set_title(f"Agent {i} View", fontsize=10, pad=5)
            ax_obs.imshow(local_map_info.map, cmap='gray', origin='lower')
            ax_obs.set_aspect('equal', adjustable='box')
            pos_cell_local = self._world_to_cell_rc(r.location, local_map_info)
            ax_obs.plot(pos_cell_local[1], pos_cell_local[0], 'o', ms=8, mfc=agent_colors(i),
                        mec='white', mew=1.5, zorder=10)
            if r.intent_seq:
                intent_world = [r.location] + r.intent_seq
                intent_cells = [self._world_to_cell_rc(pos, local_map_info) for pos in intent_world]
                ax_obs.plot([c for r_, c in intent_cells], [r_ for r_, c in intent_cells],
                            'x--', c=agent_colors(i), lw=2, ms=6, zorder=8)
            ax_obs.set_axis_off()

            ax_pred = ax_locals_pred[i]
            ax_pred.set_title(f"Agent {i} Predicted (local)", fontsize=10, pad=5)
            ax_pred.set_aspect('equal', adjustable='box')
            ax_pred.set_axis_off()

            try:
                if r.pred_mean_map_info is not None or r.pred_max_map_info is not None:
                    pred_info = r.pred_mean_map_info if r.pred_mean_map_info is not None else r.pred_max_map_info
                    pred_local = r.get_updating_map(r.location, base=pred_info)
                    belief_local = r.get_updating_map(r.location, base=r.map_info)

                    ax_pred.imshow(pred_local.map, cmap='gray', origin='lower', vmin=0, vmax=255)
                    alpha_mask = (belief_local.map == FREE) * 0.45
                    ax_pred.imshow(belief_local.map, cmap='Blues', origin='lower', alpha=alpha_mask)

                    rc = get_cell_position_from_coords(r.location, pred_local)
                    ax_pred.plot(rc[0], rc[1], 'mo', markersize=8, zorder=6)
                else:
                    ax_pred.text(0.5, 0.5, 'No prediction', ha='center', va='center', fontsize=9)
            except Exception as e:
                ax_pred.text(0.5, 0.5, f'Pred plot err:\n{e}', ha='center', va='center', fontsize=8)

        out_path = os.path.join(self.run_dir, f"t{step:04d}.png")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        self.env.frame_files.append(out_path)
