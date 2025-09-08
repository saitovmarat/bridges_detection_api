def reset_confirmation_state(cache: dict):
    keys_to_reset = [
        "bridge_detection_history",
        "arch_gap_detection_history"
    ]
    for key in keys_to_reset:
        cache.pop(key, None) 
