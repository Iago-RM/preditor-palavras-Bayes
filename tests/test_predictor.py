from src.predictor import BayesianWordPredictor


def test_predicts_word_from_academic_context():
    predictor = BayesianWordPredictor("data/corpus.txt")
    prediction = predictor.predict_next_word("O aluno estudou para a prova de")

    assert prediction["word"].lower() == "estatistica"
    assert prediction["confidence"] > 0.0


def test_unknown_context_falls_back_to_high_frequency_word():
    predictor = BayesianWordPredictor("data/corpus.txt")
    prediction = predictor.predict_next_word("palavra totalmente inexistente no corpus")

    assert prediction["word"]
    assert 0.0 <= prediction["confidence"] <= 100.0
