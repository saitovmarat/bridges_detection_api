def is_tracking_lost(
    object_key: str,
    cache: dict, 
    loss_gap: int = 9, 
    window: int = 10
) -> bool:
    
    history = cache.get(f"{object_key}_detection_history", [])
    if len(history) < window:
        return False  
    
    recent = history[-window:]
    return sum(recent) < loss_gap
