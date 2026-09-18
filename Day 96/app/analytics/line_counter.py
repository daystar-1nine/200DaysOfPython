
def has_crossed_line(previous_point, current_point, line_y):
    if previous_point is None:
        return 0 # No cross
        
    prev_y = previous_point[1]
    curr_y = current_point[1]
    
    if prev_y < line_y and curr_y >= line_y:
        return 1 # Crossed down (OUT)
    elif prev_y > line_y and curr_y <= line_y:
        return -1 # Crossed up (IN)
        
    return 0
