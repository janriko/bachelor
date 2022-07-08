import preprocessing
from tokenizing import PipelineType

if __name__ == "__main__":

    for game in preprocessing.GameType:
        print(str(game.name))
        train_dataset, eval_dataset = preprocessing.do_preprocessing(PipelineType.TEXT, game=game)
