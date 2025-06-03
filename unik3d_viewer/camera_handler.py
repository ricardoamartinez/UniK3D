import math
from pyglet.math import Mat4, Vec3
from pyglet.window import key

class Camera:
    def __init__(self, initial_position=Vec3(0, 0, 5), world_up=Vec3(0, 1, 0)):
        self.position = initial_position
        self.rotation_x = 0.0  # Pitch
        self.rotation_y = 0.0  # Yaw
        self.world_up_vector = world_up
        self.move_speed = 2.0
        self.fast_move_speed = 6.0
        self.mouse_sensitivity = 0.1
        self.zoom_speed = 0.5

        self._forward_vector = Vec3(0, 0, -1)
        self._right_vector = Vec3(1, 0, 0)
        self._up_vector = Vec3(0, 1, 0)
        self._update_vectors()

    def _update_vectors(self):
        rot_y_rad = -math.radians(self.rotation_y)
        rot_x_rad = -math.radians(self.rotation_x)

        self._forward_vector = Vec3(
            math.sin(rot_y_rad) * math.cos(rot_x_rad),
            -math.sin(rot_x_rad),
            -math.cos(rot_y_rad) * math.cos(rot_x_rad)
        ).normalize()
        self._right_vector = self.world_up_vector.cross(self._forward_vector).normalize()
        self._up_vector = self._forward_vector.cross(self._right_vector).normalize() # Or just use world_up if no roll

    def get_view_matrix(self):
        target = self.position + self._forward_vector
        return Mat4.look_at(self.position, target, self.world_up_vector)

    def update_from_input(self, dt, key_state_handler, is_mouse_exclusive):
        current_speed = self.fast_move_speed if key_state_handler[key.LSHIFT] or key_state_handler[key.RSHIFT] else self.move_speed
        move_amount = current_speed * dt

        if key_state_handler[key.W]:
            self.position += self._forward_vector * move_amount
        if key_state_handler[key.S]:
            self.position -= self._forward_vector * move_amount
        if key_state_handler[key.A]:
            self.position += self._right_vector * move_amount # Swapped A/D to match original
        if key_state_handler[key.D]:
            self.position -= self._right_vector * move_amount # Swapped A/D to match original
        if key_state_handler[key.E] or key_state_handler[key.SPACE]: # E or Space for Up
            self.position += self.world_up_vector * move_amount
        if key_state_handler[key.Q] or key_state_handler[key.LCTRL]: # Q or LCtrl for Down
            self.position -= self.world_up_vector * move_amount
        
        # If mouse is exclusive, vectors are updated via on_mouse_motion
        # If not, vectors need to be updated based on current rotation here if rotations could change otherwise.
        # However, typically mouse motion is the only thing changing rotation directly.
        if not is_mouse_exclusive:
            self._update_vectors() # Ensure vectors are up-to-date if rotation could change elsewhere

    def on_mouse_motion(self, dx, dy, is_mouse_exclusive):
        if is_mouse_exclusive:
            self.rotation_y -= dx * self.mouse_sensitivity
            self.rotation_y %= 360
            self.rotation_x += dy * self.mouse_sensitivity
            self.rotation_x = max(min(self.rotation_x, 89.9), -89.9)
            self._update_vectors()

    def on_mouse_scroll(self, scroll_y):
        self.position += self._forward_vector * scroll_y * self.zoom_speed
        # No need to update vectors, position doesn't affect them 