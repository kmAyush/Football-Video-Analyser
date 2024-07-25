import sys
sys.path.append('../')
from utils import get_center_bbox, measure_distance

class AssignPlayerBall():
    def __init__(self):
        self.max_pb_distance = 70

    def assign_ball_to_player(self, players, ball_bbox):
        ball_position = get_center_bbox(ball_bbox)

        min_distance = 99999
        assigned_player = -1

        for player_id, player in players.items():
            player_bbox = player['bounding_box']

            distance_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_position)
            distance_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_position)

            distance = min(distance_left,distance_right)

            if distance < self.max_pb_distance:
                if distance < min_distance:
                    min_distance = distance
                    assigned_player = player_id

        return assigned_player

