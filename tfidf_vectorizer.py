from typing import List
from collections import Counter
from math import log


class CountVectorizer(object):
    """
    Экземпляр класса CountVectorizer оцифровывает корпус входных текстов.

    Атрибут класса:
    __punct_signs - перечень ожидаемых знаков препинания в корпусе.

    Атрибуты экземпляра:
    input_corpus - входной корпус,
    tokenized_corpus - корпус с удаленными знаками препинания, разбитый на слова,
    dict - словарь корпуса (в виде множества),
    matrix - терм-документная матрица.
    """
    punct_signs = '''!,.?();:'"'''


    @classmethod
    def clear_punct_sentence(cls, sentence: str) -> List[str]:
        """
        Возвращает исходное предложение, очищенное от знаков пунктуации,
        в виде списка слов, приведенных к нижнему регистру.
        """
        for punct_sign in cls.punct_signs:
            sentence = sentence.replace(punct_sign, ' ')
            sentence = sentence.lower()
        return sentence.split()


    def __init__(self):
        self.input_corpus = None
        self.tokenized_corpus = None
        self.dictionary = None
        self.count_matrix = None


    def get_feature_names(self) -> List[str]:
        """
        Возвращает словарь уникальных слов из корпуса
        """
        return self.dictionary


    def _clear_punct_corpus(self, corpus: List[str]) -> List[List[str]]:
        """
        Возвращает корпус с предложениями очищенными от знаков пунктуации.
        Каждое предложение - список слов в виде строк.
        """
        return list(map(self.clear_punct_sentence, corpus))


    def fit_transform(self, corpus: List[str]) -> List[List[int]]:
        """
        Создает атрибуты векторайзера по входному корпусу.

        Возвращает терм-документную матрицу.

        """
        if corpus:
            self.input_corpus = corpus
            self.tokenized_corpus = self._clear_punct_corpus(self.input_corpus)
            self._create_dict()
            self._create_matrix()
            return self.count_matrix
        else:
            raise ValueError("Корпус должен быть не пустым!")



    def _create_matrix(self):
        """
        Создает документарную матрицу для экземпляра класса
        """
        self.count_matrix = []
        if self.tokenized_corpus is not []:
            for item in self.tokenized_corpus:
                item_cnt = Counter(item)
                row = [item_cnt[tocken] for tocken in self.dictionary]
                self.count_matrix.append(row)


    def _create_dict(self):
        """
        Создает словарь токенов для экземпляра класса.
        Сортирует словарь лексикографически.
        """
        self.dictionary = set()
        self.dictionary.update(*self.tokenized_corpus)
        self.dictionary = list(self.dictionary)
        self.dictionary.sort()


class TfidfTransformer():
    """
    Содержит методы вычисления tf, idf, tf-idf.
    """
    @staticmethod
    def rnd_prod(a: float, b: float, n: int=3):
        """
        Возвращает округленное произведение двух чисел до требуемого
        числа знаков после запятой.
        """
        return round(a * b, n)


    def __init__(self):
        self.row_length = None
        self.tfidf_matrix = None


    def  tf_transform(self, count_matrix: List[List[int]]) -> List[List[float]]:
        """
        Возвращает tf матрицу по заданной терм-документной матрице.
        """
        tf_matrix = []
        for row in count_matrix:
            row_sum = sum(row)
            tf_matrix.append([x / row_sum for x in row])
        return tf_matrix


    def  idf_transform(self, count_matrix: List[List[int]]) -> List[List[float]]:
        """
        Возвращает idf матрицу по заданной терм-документной матрице.
        """
        idf_matrix = []
        denom = len(count_matrix) + 1
        self.row_length = len(count_matrix[0])
        for icol in range(self.row_length):
            divr = sum([row[icol] != 0 for row in count_matrix]) + 1
            idf_matrix.append(log(denom / divr) + 1)
        return idf_matrix


    def fit_transform(self, count_matrix: List[List[int]]) -> List[List[float]]:
        """
        Возвращает tf-idf матрицу по заданной терм-документной матрице.
        """
        tf_matrix = self.tf_transform(count_matrix)
        idf_matrix = self.idf_transform(count_matrix)
        tfidf_matrix = []
        for row in tf_matrix:
            tfidf_matrix.append([self.rnd_prod(row[i], idf_matrix[i])
                                 for i in range(self.row_length)])
        return tfidf_matrix


class TfidfVectorizer(CountVectorizer, TfidfTransformer):
    """
       Наследует от CountVectorizer.
       Дополнен: атрибутом - матрица tf-idf и
                 методом, возвращающим матрицу tf-idf.
    """
    def __init__(self):
        super().__init__()
        self.tfidf_matrix = None


    def fit_transform(self, corpus: List[List[str]]) -> List[List[float]]:
        """
        Возвращает tf-idf матрицу по заданному корпусу.
        """
        super().fit_transform(corpus)
        self.tfidf_matrix = TfidfTransformer.fit_transform(self,
                                                           self.count_matrix)
        return self.tfidf_matrix


if __name__ == '__main__':
    corpus = ['Crock Pot Pasta Never boil pasta again',
              'Pasta Pomodoro Fresh ingredients Parmesan to taste']
    print('Тестовый корпус:')
    for row in corpus:
        print(row)
    print()
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(corpus)
    dictionary = tfidf.get_feature_names()
    assert dictionary == ['again', 'boil', 'crock', 'fresh', 'ingredients',
                          'never', 'parmesan', 'pasta', 'pomodoro', 'pot',
                          'taste', 'to'],\
                         'Словарь построен неверно!'
    print('Выводим словарь:')
    print(dictionary)
    print()
    assert tfidf_matrix == [[0.201, 0.201, 0.201, 0.0,   0.0,   0.201, 0.0,
                             0.286, 0.0,   0.201, 0.0,   0.0],
                            [0.0,   0.0,   0.0,   0.201, 0.201, 0.0,   0.201,
                             0.143, 0.201, 0.0,   0.201, 0.201]],\
                           'Ошибка в tfidf матрице!'
    print('Выводим tfidf матрицу:')
    for row in tfidf_matrix:
        print(row)