from src.predictor import BayesianWordPredictor


def main() -> None:
    predictor = BayesianWordPredictor("data/corpus.txt")

    frases = [
        "O aluno estudou para a prova de",
        "Na disciplina de",
        "A aula de estatistica explicou",
        "Em todas as aulas de",
    ]

    for frase in frases:
        resultado = predictor.predict_next_word(frase)
        print(f"Frase: {frase}")
        print(f"Sugestão: {resultado['word']}, Confiança: {resultado['confidence']:.2f}%")
        print("-" * 40)


if __name__ == "__main__":
    main()
