"""just a quick script to fix that annoying suggestion panel bug"""
import re

path = r'c:\Users\laves\Agent Sheild\ui\dashboard.py'
content = open(path, 'r', encoding='utf-8').read()




START_MARKER = "            async def analyse_input():"
END_MARKER   = "                    refresh_wellness_ui()\n\n"

start_idx = content.find(START_MARKER)
end_idx   = content.find(END_MARKER, start_idx) + len(END_MARKER)

if start_idx == -1:
    print("ERROR: Could not find analyse_input start marker")
    exit(1)
if end_idx == -1:
    print("ERROR: Could not find end marker")
    exit(1)

old_block = content[start_idx:end_idx]
print(f"Found block ({len(old_block)} chars). Replacing...")

new_block = '''\
            async def analyse_input():
                text = text_input.value.strip()
                if not text or not state.call_active:
                    if not state.call_active:
                        ui.notify("Start a call first!", color="negative")
                    return

                text_input.value = ""
                add_transcript_entry("customer", text)

                suggestion_label.set_content(
                    '<div class="suggestion-box" style="color:#64748b;font-style:italic;">'
                    'Analysing query...'
                    '</div>'
                )

                data = await api_post("/api/calls/analyse-text", {
                    "session_id": state.session_id,
                    "text": text,
                    "speaker": "customer"
                })

                if not data:
                    suggestion_label.set_content(
                        '<div class="suggestion-box" style="color:#ef4444;">'
                        '<strong>API Error</strong><br/>'
                        'Could not reach the backend on port 8080.'
                        '</div>'
                    )
                    return

                state.toxicity_score = data.get("toxicity_score", 0.0)
                state.toxicity_level = data.get("toxicity_level", "safe")
                state.alert_message  = data.get("alert_message") or ""
                refresh_toxicity_ui()

                suggestion = data.get("ai_suggestion") or ""
                if suggestion and suggestion.strip():
                    formatted = (
                        suggestion
                        .replace("\\n", "<br/>")
                        .replace("* ", "<br/>&bull; ")
                    )
                    suggestion_label.set_content(
                        f'<div class="suggestion-box">{formatted}</div>'
                    )
                else:
                    suggestion_label.set_content(
                        '<div class="suggestion-box">'
                        '<span style="color:#eab308;font-weight:600;">No KB match found</span>'
                        '<br/><br/>'
                        '<span style="color:#94a3b8;">'
                        'No policy found for this query.<br/>'
                        'Use your judgement or search the Knowledge Base panel.'
                        '</span>'
                        '</div>'
                    )

                wellness_data = await api_get(f"/api/wellness/{state.agent_id}/status")
                if wellness_data:
                    state.wellness_score = wellness_data["wellness_score"]
                    state.stress_level   = wellness_data["stress_level"]
                    refresh_wellness_ui()

'''

content = content[:start_idx] + new_block + content[end_idx:]
open(path, 'w', encoding='utf-8').write(content)
print("SUCCESS: dashboard.py patched correctly.")
print(f"New function length: {len(new_block)} chars")
