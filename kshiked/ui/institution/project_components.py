import streamlit as st
import pandas as pd
from typing import Dict, List
from kshiked.ui.institution.backend.project_manager import ProjectManager
from kshiked.ui.theme import LIGHT_THEME

def render_project_wizard(all_baskets: Dict[int, str], current_basket_id: int):
  """
  Renders a multi-step wizard to create a structured operational project.
  Can be included in both Admin and Executive views.
  """
  if "proj_wizard_step" not in st.session_state:
    st.session_state.proj_wizard_step = 1

  if "proj_data" not in st.session_state:
    st.session_state.proj_data = {
      "title": "",
      "description": "",
      "severity": 5.0,
      "participants": [],
      "vision": "",
      "objectives": [],
      "milestones": [],
      "outcomes": []
    }

  st.write("#### Launch Structured Operational Project")
  st.write("Define vision, objectives, milestones, and expected outcomes.")
  
  # Progress visualization
  steps = ["1. Basics", "2. Vision & Objectives", "3. Milestones", "4. Outcomes", "5. Review"]
  st.write(" > ".join([f"**{s}**" if i+1 == st.session_state.proj_wizard_step else s for i, s in enumerate(steps)]))
  st.write("---")

  # Step 1: Basics
  if st.session_state.proj_wizard_step == 1:
    st.session_state.proj_data["title"] = st.text_input("Project Name", st.session_state.proj_data["title"])
    st.session_state.proj_data["description"] = st.text_area("Initial SitRep / Description", st.session_state.proj_data["description"])
    st.session_state.proj_data["severity"] = st.slider("Assigned Severity", 1.0, 10.0, st.session_state.proj_data["severity"], 0.5)
    
    other_baskets = {k: v for k, v in all_baskets.items() if k != current_basket_id}
    st.session_state.proj_data["participants"] = st.multiselect(
      "Invite Sector Baskets", 
      options=list(other_baskets.keys()), 
      default=st.session_state.proj_data["participants"],
      format_func=lambda x: other_baskets[x]
    )
    
    if st.button("Next: Vision & Objectives >"):
      if st.session_state.proj_data["title"] and st.session_state.proj_data["participants"]:
        st.session_state.proj_wizard_step = 2
        st.rerun()
      else:
        st.error("Title and at least one invited participant are required.")

  # Step 2: Vision & Objectives
  elif st.session_state.proj_wizard_step == 2:
    st.session_state.proj_data["vision"] = st.text_area("Project Vision", st.session_state.proj_data["vision"], help="Overarching goal of this project")
    st.write("##### Objectives")
    
    for i, obj in enumerate(st.session_state.proj_data["objectives"]):
      with st.expander(f"Objective {i+1}: {obj.get('title', 'New')}", expanded=True):
        obj["title"] = st.text_input("Objective Title", obj.get("title", ""), key=f"obj_title_{i}")
        obj["description"] = st.text_area("Description", obj.get("description", ""), key=f"obj_desc_{i}")
        obj["success_metric"] = st.text_input("Success Metric (How will we know?)", obj.get("success_metric", ""), key=f"obj_met_{i}")
        
    if st.button("+ Add Objective"):
      if len(st.session_state.proj_data["objectives"]) < 10:
        st.session_state.proj_data["objectives"].append({"title": "", "description": "", "success_metric": ""})
        st.rerun()
      else:
        st.warning("Max 10 objectives allowed.")
        
    col1, col2 = st.columns(2)
    with col1:
      if st.button("< Back"):
        st.session_state.proj_wizard_step = 1
        st.rerun()
    with col2:
      if st.button("Next: Milestones >"):
        if not st.session_state.proj_data["vision"]:
          st.error("Project Vision is required.")
        elif len(st.session_state.proj_data["objectives"]) < 1:
          st.error("At least 1 Objective is required.")
        elif not all(o.get('title') and o.get('success_metric') for o in st.session_state.proj_data["objectives"]):
          st.error("All Objectives must have a Title and Success Metric.")
        else:
          st.session_state.proj_wizard_step = 3
          st.rerun()

  # Step 3: Milestones
  elif st.session_state.proj_wizard_step == 3:
    st.write("##### Project Milestones")
    obj_options = [o.get("title", f"Obj {i}") for i, o in enumerate(st.session_state.proj_data["objectives"])]
    
    for i, ms in enumerate(st.session_state.proj_data["milestones"]):
      with st.expander(f"Milestone {i+1}: {ms.get('title', 'New')}", expanded=True):
        ms["title"] = st.text_input("Title", ms.get("title", ""), key=f"ms_title_{i}")
        ms["description"] = st.text_area("Description", ms.get("description", ""), key=f"ms_desc_{i}")
        ms["assigned_to"] = st.selectbox("Assigned To Sector", ["Any"] + [all_baskets.get(b, "Unknown") for b in [current_basket_id] + st.session_state.proj_data["participants"]], key=f"ms_assign_{i}")
        # Convert dates to/from timestamp for simplicity, but simple text input works as placeholder for now, or Streamlit date_input
        ms["due_date"] = st.number_input("Due in (Days from now)", min_value=1, value=ms.get("due_date", 14) if isinstance(ms.get("due_date"), int) else 14, key=f"ms_due_{i}")
        ms["linked_objectives"] = st.multiselect("Linked Objectives", options=obj_options, default=ms.get("linked_objectives", []), key=f"ms_link_{i}")
        
    if st.button("+ Add Milestone"):
      st.session_state.proj_data["milestones"].append({"title": "", "description": "", "assigned_to": "Any", "due_date": 14, "linked_objectives": []})
      st.rerun()
      
    col1, col2 = st.columns(2)
    with col1:
      if st.button("< Back"):
        st.session_state.proj_wizard_step = 2
        st.rerun()
    with col2:
      if st.button("Next: Expected Outcomes >"):
        if len(st.session_state.proj_data["milestones"]) < 2:
          st.error("At least 2 Milestones are required.")
        elif not all(m.get('title') for m in st.session_state.proj_data["milestones"]):
          st.error("All Milestones require a Title.")
        else:
          st.session_state.proj_wizard_step = 4
          st.rerun()

  # Step 4: Expected Outcomes
  elif st.session_state.proj_wizard_step == 4:
    st.write("##### Expected Deliverables / Outcomes")
    for i, out in enumerate(st.session_state.proj_data["outcomes"]):
      with st.container(border=True):
        out["title"] = st.text_input("Title", out.get("title", ""), key=f"out_title_{i}")
        out["description"] = st.text_input("Description", out.get("description", ""), key=f"out_desc_{i}")
        out["measurable_target"] = st.text_input("Measurable Target", out.get("measurable_target", ""), key=f"out_target_{i}")
        
    if st.button("+ Add Outcome"):
      st.session_state.proj_data["outcomes"].append({"title": "", "description": "", "measurable_target": ""})
      st.rerun()

    col1, col2 = st.columns(2)
    with col1:
      if st.button("< Back"):
        st.session_state.proj_wizard_step = 3
        st.rerun()
    with col2:
      if st.button("Next: Review >"):
        if len(st.session_state.proj_data["outcomes"]) < 1:
          st.error("At least 1 Outcome is required.")
        elif not all(o.get('title') and o.get('measurable_target') for o in st.session_state.proj_data["outcomes"]):
          st.error("All Outcomes require a Title and Measurable Target.")
        else:
          st.session_state.proj_wizard_step = 5
          st.rerun()

  # Step 5: Review & Submit
  elif st.session_state.proj_wizard_step == 5:
    data = st.session_state.proj_data
    st.write(f"**Project:** {data['title']}")
    st.write(f"**Severity:** {data['severity']}")
    st.write(f"**Participants:** {[all_baskets.get(p) for p in data['participants']]}")
    st.write(f"**Vision:** {data['vision']}")
    st.write(f"**Total Objectives:** {len(data['objectives'])}")
    st.write(f"**Total Milestones:** {len(data['milestones'])}")
    st.write(f"**Total Outcomes:** {len(data['outcomes'])}")
    
    col1, col2 = st.columns(2)
    with col1:
      if st.button("< Back"):
        st.session_state.proj_wizard_step = 4
        st.rerun()
    with col2:
      if st.button("Initialize Shared Space", type="primary"):
        # Append self to participants
        final_participants = [current_basket_id] + data["participants"]
        
        # Transform due_date from Days offset into actual timestamps
        import time
        milestones_payload = []
        for ms in data["milestones"]:
          ms_copy = ms.copy()
          ms_copy["due_date"] = time.time() + (ms_copy["due_date"] * 86400)
          milestones_payload.append(ms_copy)
        
        ProjectManager.create_project(
          title=data["title"],
          description=data["description"],
          severity=data["severity"],
          participant_basket_ids=final_participants,
          vision=data["vision"],
          objectives=data["objectives"],
          milestones=milestones_payload,
          outcomes=data["outcomes"]
        )
        
        st.success(f"Operational Project '{data['title']}' launched successfully.")
        # Reset state
        st.session_state.proj_wizard_step = 1
        st.session_state.proj_data = {
          "title": "", "description": "", "severity": 5.0, "participants": [],
          "vision": "", "objectives": [], "milestones": [], "outcomes": []
        }
        
        import time as dt
        dt.sleep(1)
        st.rerun()

@st.dialog("End Project & Post-Mortem Review")
def render_post_mortem_dialog(project_id: int):
  st.write("Evaluate the project outcomes and provide a final verdict.")
  
  verdict = st.selectbox("Overall Project Verdict", ["Successful", "Partially Successful", "Unsuccessful"])
  justification = st.text_area("Justification (Required)", help="Why did you choose this verdict?")
  lessons = st.text_area("Lessons Learned", help="What can be improved for future operational projects?")
  
  if st.button("Submit Post-Mortem & Archive Project", type="primary"):
    if justification:
      import time
      from kshiked.ui.institution.backend.database import get_connection
      with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
          "INSERT INTO project_post_mortem (project_id, verdict, justification, lessons_learned) VALUES (?, ?, ?, ?)",
          (project_id, verdict, justification, lessons)
        )
        # Auto-transition to RECOVERY phase or ARCHIVED status
        cur.execute("UPDATE operational_projects SET status = 'ARCHIVED' WHERE id = ?", (project_id,))
        
        # We could ideally update the outcome achievements here too, but storing the verdict is the core requirement.
        conn.commit()
      st.success("Project archived successfully.")
      import time as dt
      dt.sleep(1)
      st.rerun()
    else:
      st.error("Justification is required.")

def render_project_overview(proj: Dict, role: str, current_basket_id: int, all_baskets: Dict[int, str]):
  """
  Renders the health overview of a single project, including:
  - Progress bars (Objectives / Milestones)
  - Milestone Status Toggle for Spokes
  - Activity Log
  - Post-Mortem flow
  """
  import time
  now = time.time()
  
  # Calculate Progress
  milestones = proj.get("milestones", [])
  objectives = proj.get("objectives", [])
  
  total_ms = len(milestones)
  completed_ms = sum(1 for m in milestones if m["status"] == "COMPLETED")
  ms_progress = completed_ms / total_ms if total_ms > 0 else 0.0
  
  # Objectives are achieved if all linked milestones are completed. 
  # (Or we can just evaluate it conceptually. Let's do a simple ratio of completed milestones for now).
  
  st.markdown(f"**Vision:** {proj.get('vision', 'No vision defined.')}")
  
  # Progress Bars
  c1, c2 = st.columns(2)
  with c1:
    st.write("Milestone Progress")
    st.progress(ms_progress, text=f"{completed_ms} of {total_ms} Milestones Completed")
  with c2:
    st.write("Health Status")
    overdue_count = sum(1 for m in milestones if m["due_date"] < now and m["status"] != "COMPLETED")
    blocked_count = sum(1 for m in milestones if m["status"] == "BLOCKED")
    
    if blocked_count > 0:
      st.error(f" {blocked_count} Milestone(s) BLOCKED")
    elif overdue_count > 0:
      st.warning(f" {overdue_count} Milestone(s) OVERDUE")
    else:
      st.success(" On Track")

  st.write("---")
  
  t_obj, t_ms, t_out, t_log = st.tabs(["Objectives", "Milestones", "Outcomes", "Activity Log"])
  
  with t_obj:
    for obj in objectives:
      st.markdown(f"**{obj['title']}** - {obj['description']}")
      st.caption(f"Success Metric: {obj['success_metric']}")
      
  with t_ms:
    for ms in milestones:
      is_overdue = ms["due_date"] < now and ms["status"] != "COMPLETED"
      border_color = "red" if is_overdue else "green" if ms["status"] == "COMPLETED" else "orange" if ms["status"] == "BLOCKED" else "gray"
      
      with st.container(border=True):
        st.markdown(f"**{ms['title']}**")
        st.write(ms["description"])
        st.caption(f"Assigned to: {ms['assigned_to']}")
        
        # Check if current user can edit this milestone
        current_basket_name = all_baskets.get(current_basket_id, "Unknown")
        can_edit = role in ["Executive", "Admin"] or ms["assigned_to"] in ["Any", current_basket_name]
        
        col_st, col_act = st.columns([2, 1])
        with col_st:
          if can_edit:
            new_status = st.selectbox(
              "Status", 
              ["NOT_STARTED", "IN_PROGRESS", "BLOCKED", "COMPLETED"], 
              index=["NOT_STARTED", "IN_PROGRESS", "BLOCKED", "COMPLETED"].index(ms["status"]),
              key=f"status_{ms['id']}"
            )
            note = st.text_input("Status Note (Optional)", key=f"note_{ms['id']}")
            if st.button("Update Status", key=f"btn_upd_{ms['id']}"):
              ProjectManager.update_milestone_status(
                milestone_id=ms['id'], 
                new_status=new_status, 
                changed_by=f"{role} ({current_basket_name})", 
                note=note
              )
              st.rerun()
          else:
            st.info(f"Status: {ms['status']}")
            
        with col_act:
          import datetime
          due_date_str = datetime.datetime.fromtimestamp(ms["due_date"]).strftime('%Y-%m-%d')
          if is_overdue:
            st.error(f"Due: {due_date_str}")
          else:
            st.write(f"Due: {due_date_str}")

  with t_out:
    for out in proj.get("outcomes", []):
      st.markdown(f"- **{out['title']}**: {out['measurable_target']}")
      
  with t_log:
    logs = proj.get("activity_log", [])
    if not logs:
      st.info("No milestone activity recorded yet.")
    for log in logs:
      import datetime
      ts = datetime.datetime.fromtimestamp(log['timestamp']).strftime('%m/%d %H:%M')
      st.caption(f"[{ts}] **{log['changed_by']}** changed status from `{log['old_status']}` to `{log['new_status']}`.")
      if log.get("note"):
        st.write(f"> {log['note']}")

  # End Project Flow
  if role in ["Executive", "Admin"] and proj["status"] != "ARCHIVED":
    st.write("---")
    if st.button("End Project & Submit Post-Mortem", type="secondary", key=f"end_proj_{proj['id']}"):
      render_post_mortem_dialog(proj['id'])
