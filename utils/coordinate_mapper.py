class CoordinateMapper:
    def __init__(self, room):
        self.room = room

    def transform_width(self, value = 0.0):
        env_width = abs(self.room["min_width"]) + abs(self.room["max_width"])
        env_width_WP = abs(self.room["min_width_WP"]) + abs(self.room["max_width_WP"])
        return (((value - self.room["min_width_WP"]) / env_width_WP) * env_width) + self.room["min_width"]
    
    def transform_height(self, value = 0.0):
        newValue = -value + self.room["max_height_WP"] - self.room["height_offset"]
        return max(0, min(newValue, self.room["max_height"])) # clamp height value
    
    def rescale(self, x = 0.0, y = 0.0, z = 0.0):
        return (self.transform_width(x) - abs(self.room["world_x_origin"])),
        (self.transform_height(y)),
        (-(abs(z) + abs(self.room["world_z_origin"]) + self.room["backwall_distance"])),