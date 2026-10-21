"""Trusted recovery of exact sampled interactions into the native session cache.

This has no model-visible tool or HTTP route. Callers must first validate durable
wire evidence and prove that the source did not already produce eligible rows.
"""


def restore_native_cache(session, records):
    if session.active_completions:
        raise ValueError("refuse restoration into a nonempty session")
    for cid, interaction, binding in records:
        response = interaction.model_response
        if (
            binding["session_id"] != session.session_id
            or cid in session.active_completions
            or response is None
            or not response.output_tokens
            or len(response.output_tokens) != len(response.output_logprobs)
            or len(response.output_tokens) != len(response.output_versions)
            or interaction.interaction_id != cid
        ):
            raise ValueError("sampled recovery identity/shape mismatch")
        session.active_completions[cid] = interaction
    if list(session.active_completions) != [r[0] for r in records]:
        raise ValueError("sampled recovery ordering mismatch")
    session.update_last_access()
