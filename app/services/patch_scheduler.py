import os

file_path = '/Users/tejeshwarcdr/Documents/Documents - Tejeshwar’s MacBook Pro/ShedForge/ShedForge/backend/app/services/evolution_scheduler.py'

new_method_code = """    def _constructive_individual(
        self,
        *,
        randomized: bool,
        rcl_alpha: float = 0.05,
    ) -> list[int]:
        genes = [0] * len(self.block_requests)
        
        # Tracking state locally for the constructive build
        selected_options: dict[int, PlacementOption] = {}
        room_occ: dict[tuple[str, int, str], list[int]] = defaultdict(list)
        faculty_occ: dict[tuple[str, int, str], list[int]] = defaultdict(list)
        section_occ: dict[tuple[str, int, str], list[int]] = defaultdict(list)
        faculty_minutes: dict[str, int] = {}
        section_slot_keys: dict[str, set[tuple[str, int]]] = defaultdict(set)
        
        # Parallel lab tracking
        lab_baseline_batch_by_group: dict[tuple[str, str, str, int], str] = {}
        lab_baseline_signatures_by_group: dict[tuple[str, str, str, int], list[tuple[str, int]]] = defaultdict(list)
        lab_signature_usage_by_group_batch: dict[tuple[tuple[str, str, str, int], str], Counter[tuple[str, int]]] = defaultdict(Counter)

        sorted_indices = self._request_priority_order()
        
        for req_index in sorted_indices:
            req = self.block_requests[req_index]

            # 1. Respect pre-fixed genes (e.g. from partial solutions or locks)
            if req.request_id in self.fixed_genes:
                chosen_index = self.fixed_genes[req.request_id]
                genes[req_index] = chosen_index
                self._record_selection(
                    req_index, 
                    chosen_index, 
                    selected_options, 
                    room_occ, 
                    faculty_occ, 
                    section_occ, 
                    faculty_minutes, 
                    section_slot_keys,
                    lab_baseline_batch_by_group,
                    lab_baseline_signatures_by_group,
                    lab_signature_usage_by_group_batch
                )
                continue

            # 2. Determine Candidate Options
            all_candidate_indices = list(range(len(req.options)))
            if randomized and len(all_candidate_indices) > 60:
                 self.random.shuffle(all_candidate_indices)
                 all_candidate_indices = all_candidate_indices[:60]

            # 3. Filter for Hard Feasibility
            feasible_indices = []
            for opt_idx in all_candidate_indices:
                if self._is_immediately_conflict_free(
                    req_index=req_index,
                    option_index=opt_idx,
                    selected_options=selected_options,
                    room_occ=room_occ,
                    faculty_occ=faculty_occ,
                    section_occ=section_occ,
                    faculty_minutes=faculty_minutes,
                    section_slot_keys=section_slot_keys,
                ):
                    feasible_indices.append(opt_idx)
            
            # Fallback to soft failure if no feasible options
            candidates_to_score = feasible_indices if feasible_indices else all_candidate_indices
            
            # 4. Score Candidates (Best-Fit)
            scored_candidates: list[tuple[float, int]] = []
            
            for opt_idx in candidates_to_score:
                hard_score, soft_score = self._incremental_option_penalty(
                    req_index=req_index,
                    option_index=opt_idx,
                    selected_options=selected_options,
                    room_occ=room_occ,
                    faculty_occ=faculty_occ,
                    section_occ=section_occ,
                    elective_occ=defaultdict(list),
                    faculty_minutes=faculty_minutes,
                    section_slot_keys=section_slot_keys,
                )
                
                room = self.rooms[req.options[opt_idx].room_id]
                capacity_waste = 0.0
                if room.capacity >= req.student_count:
                    capacity_waste = (room.capacity - req.student_count) / max(1, room.capacity)
                
                # Heuristic Weighting
                final_score = (hard_score * 10000.0) + soft_score + (capacity_waste * 0.5)
                scored_candidates.append((final_score, opt_idx))
            
            # 5. Select Best
            scored_candidates.sort(key=lambda x: x[0])
            
            chosen_index = -1
            if not scored_candidates:
                chosen_index = 0
            elif randomized and rcl_alpha > 0 and len(scored_candidates) > 1:
                best_score = scored_candidates[0][0]
                threshold = best_score + (abs(best_score) * rcl_alpha) + 1.0
                rcl = [idx for score, idx in scored_candidates if score <= threshold]
                chosen_index = self.random.choice(rcl)
            else:
                chosen_index = scored_candidates[0][1]
                
            genes[req_index] = chosen_index
            
            # 6. Update State
            self._record_selection(
                req_index, 
                chosen_index, 
                selected_options, 
                room_occ, 
                faculty_occ, 
                section_occ, 
                faculty_minutes, 
                section_slot_keys,
                lab_baseline_batch_by_group,
                lab_baseline_signatures_by_group,
                lab_signature_usage_by_group_batch
            )

        return genes

    def _record_selection(
        self,
        req_index: int,
        option_index: int,
        selected_options: dict[int, PlacementOption],
        room_occ: dict,
        faculty_occ: dict,
        section_occ: dict,
        faculty_minutes: dict,
        section_slot_keys: dict,
        lab_baseline_batch_by_group: dict,
        lab_baseline_signatures_by_group: dict,
        lab_signature_usage_by_group_batch: dict
    ):
        req = self.block_requests[req_index]
        option = req.options[option_index]
        selected_options[req_index] = option
        
        if req.is_lab:
            group_key = self._parallel_lab_group_key(req)
            if group_key and req.batch:
                lab_baseline_batch_by_group.setdefault(group_key, req.batch)
                signature = self._parallel_lab_signature(option)
                if req.batch == lab_baseline_batch_by_group[group_key]:
                    lab_baseline_signatures_by_group[group_key].append(signature)
                lab_signature_usage_by_group_batch[(group_key, req.batch)][signature] += 1

        for offset in range(req.block_size):
            slot_idx = option.start_index + offset
            room_key = (option.day, slot_idx, option.room_id)
            faculty_key = (option.day, slot_idx, option.faculty_id)
            section_key = (option.day, slot_idx, req.section)
            
            room_occ[room_key].append(req_index)
            faculty_occ[faculty_key].append(req_index)
            section_occ[section_key].append(req_index)
            section_slot_keys[req.section].add((option.day, slot_idx))
        
        added_minutes = req.block_size * self.schedule_policy.period_minutes
        faculty_minutes[option.faculty_id] = faculty_minutes.get(option.faculty_id, 0) + added_minutes
"""

with open(file_path, 'r') as f:
    lines = f.readlines()

# Lines are 1-indexed in view_file but 0-indexed in list
# Replace 2972 to 3229 (inclusive 1-based)
# indices: 2971 to 3229 (3229 because slice end is exclusive, wait)
# 1-based: 2972. Index: 2971.
# 1-based: 3229. Index: 3228.
# Slice: [2971:3229] (since 3229 is excluded, it goes up to 3228)

start_idx = 2971
end_idx = 3229

new_lines = lines[:start_idx] + [new_method_code + "\n"] + lines[end_idx:]

with open(file_path, 'w') as f:
    f.writelines(new_lines)

print("Successfully patched evolution_scheduler.py")
