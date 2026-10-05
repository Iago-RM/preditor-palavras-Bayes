# Sistema Preditivo de Palavras com Teorema de Bayes

Este projeto implementa um sistema de autocompletação de texto com base em modelos de n-grams e no Teorema de Bayes. A lógica principal calcula a probabilidade de uma palavra seguir um contexto, usando suavização de Laplace para evitar probabilidades nulas quando o contexto não aparece no corpus.

## Objetivo

Dado um contexto textual, o sistema sugere a palavra mais provável como próxima ocorrência e informa a porcentagem de confiança da previsão.

## Fórmula utilizada

A probabilidade condicional é calculada pela forma clássica:

P(Wn | Contexto) = P(Contexto | Wn) * P(Wn) / P(Contexto)

Na prática, a implementação usa a estimativa de n-grams:

P(w | h) = (count(h, w) + α) / (count(h) + α * |V|)

onde:
- h é o histórico de palavras (contexto);
- w é a palavra candidata;
- α é o parâmetro de suavização de Laplace;
- |V| é o tamanho do vocabulário.

## Estrutura do projeto

- src/preprocessor.py: normalização e tokenização do texto.
- src/ngram_model.py: contagem de n-grams e cálculo de probabilidades.
- src/predictor.py: interface pública do preditor.
- data/corpus.txt: corpus utilizado para treinamento.
- main.py: exemplo de execução do sistema.
- tests/test_predictor.py: testes automatizados de comportamento.

## Como executar

1. Abra o terminal na raiz do projeto.
2. Instale as dependências, se houver necessidade (o projeto usa apenas Python padrão).
3. Execute:

```bash
python main.py
```

## Como testar

```bash
python -m pytest -q
```

## Exemplo de saída

```text
Frase: O aluno estudou para a prova de
Sugestão: estatistica, Confiança: 55.00%
```

## Observações de design

- O corpus foi construído com frases acadêmicas para reforçar o contexto de provas e estatística.
- A tokenização considera letras acentuadas e números, além de remover pontuação irrelevante.
- A estratégia de backoff prioriza trigramas, depois bigramas e, por fim, unigramas, garantindo melhor desempenho para contextos mais específicos.
- Quando o contexto é raro ou inexistente, o sistema recorre à distribuição geral do vocabulário com suavização.

## Limitações

Como trata-se de um modelo estatístico simples, o sistema depende muito da qualidade e da variedade do corpus. Em textos mais heterogêneos, a precisão pode diminuir, mas a abordagem permanece robusta e facilmente extensível para corpora maiores.
