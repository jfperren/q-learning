from collections import namedtuple


DiscreteQLearningTrainingConfig = namedtuple(
    'TrainingConfig',
    ['learning_rate', 'discount', 'episodes']
)
