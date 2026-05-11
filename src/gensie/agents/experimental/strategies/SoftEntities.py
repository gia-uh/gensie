from .FixedEntities import FixedEntities

class SoftEntities(FixedEntities):

    def estimate(self, task, fields):
        super().estimate(task, fields)