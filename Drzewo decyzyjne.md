# Drzewo decyzyjne

## Biblioteki:

```python
import numpy as np
from collections import Counter
```

- **Kod tego gościa z filmu:**
    
    ### Node class
    
    ```python
    class Node():
        def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
            ''' constructor ''' 
            
            # for decision node
            self.feature_index = feature_index
            self.threshold = threshold
            self.left = left
            self.right = right
            self.info_gain = info_gain
            
            # for leaf node
            self.value = value
    ```
    
    ### Tree class
    
    ```python
    class DecisionTreeClassifier():
        def __init__(self, min_samples_split=2, max_depth=2):
            ''' constructor '''
            
            # initialize the root of the tree 
            self.root = None
            
            # stopping conditions
            self.min_samples_split = min_samples_split
            self.max_depth = max_depth
            
        def build_tree(self, dataset, curr_depth=0):
            ''' recursive function to build the tree ''' 
            
            X, Y = dataset[:,:-1], dataset[:,-1]
            num_samples, num_features = np.shape(X)
            
            # split until stopping conditions are met
            if num_samples>=self.min_samples_split and curr_depth<=self.max_depth:
                # find the best split
                best_split = self.get_best_split(dataset, num_samples, num_features)
                # check if information gain is positive
                if best_split["info_gain"]>0:
                    # recur left
                    left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                    # recur right
                    right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                    # return decision node
                    return Node(best_split["feature_index"], best_split["threshold"], 
                                left_subtree, right_subtree, best_split["info_gain"])
            
            # compute leaf node
            leaf_value = self.calculate_leaf_value(Y)
            # return leaf node
            return Node(value=leaf_value)
        
        def get_best_split(self, dataset, num_samples, num_features):
            ''' function to find the best split '''
            
            # dictionary to store the best split
            best_split = {}
            max_info_gain = -float("inf")
            
            # loop over all the features
            for feature_index in range(num_features):
                feature_values = dataset[:, feature_index]
                possible_thresholds = np.unique(feature_values)
                # loop over all the feature values present in the data
                for threshold in possible_thresholds:
                    # get current split
                    dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                    # check if childs are not null
                    if len(dataset_left)>0 and len(dataset_right)>0:
                        y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                        # compute information gain
                        curr_info_gain = self.information_gain(y, left_y, right_y, "gini")
                        # update the best split if needed
                        if curr_info_gain>max_info_gain:
                            best_split["feature_index"] = feature_index
                            best_split["threshold"] = threshold
                            best_split["dataset_left"] = dataset_left
                            best_split["dataset_right"] = dataset_right
                            best_split["info_gain"] = curr_info_gain
                            max_info_gain = curr_info_gain
                            
            # return best split
            return best_split
        
        def split(self, dataset, feature_index, threshold):
            ''' function to split the data '''
            
            dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
            dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
            return dataset_left, dataset_right
        
        def information_gain(self, parent, l_child, r_child, mode="entropy"):
            ''' function to compute information gain '''
            
            weight_l = len(l_child) / len(parent)
            weight_r = len(r_child) / len(parent)
            if mode=="gini":
                gain = self.gini_index(parent) - (weight_l*self.gini_index(l_child) + weight_r*self.gini_index(r_child))
            else:
                gain = self.entropy(parent) - (weight_l*self.entropy(l_child) + weight_r*self.entropy(r_child))
            return gain
        
        def entropy(self, y):
            ''' function to compute entropy '''
            
            class_labels = np.unique(y)
            entropy = 0
            for cls in class_labels:
                p_cls = len(y[y == cls]) / len(y)
                entropy += -p_cls * np.log2(p_cls)
            return entropy
        
        def gini_index(self, y):
            ''' function to compute gini index '''
            
            class_labels = np.unique(y)
            gini = 0
            for cls in class_labels:
                p_cls = len(y[y == cls]) / len(y)
                gini += p_cls**2
            return 1 - gini
            
        def calculate_leaf_value(self, Y):
            ''' function to compute leaf node '''
            
            Y = list(Y)
            return max(Y, key=Y.count)
        
        def print_tree(self, tree=None, indent=" "):
            ''' function to print the tree '''
            
            if not tree:
                tree = self.root
    
            if tree.value is not None:
                print(tree.value)
    
            else:
                print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
                print("%sleft:" % (indent), end="")
                self.print_tree(tree.left, indent + indent)
                print("%sright:" % (indent), end="")
                self.print_tree(tree.right, indent + indent)
        
        def fit(self, X, Y):
            ''' function to train the tree '''
            
            dataset = np.concatenate((X, Y), axis=1)
            self.root = self.build_tree(dataset)
        
        def predict(self, X):
            ''' function to predict new dataset '''
            
            preditions = [self.make_prediction(x, self.root) for x in X]
            return preditions
        
        def make_prediction(self, x, tree):
            ''' function to predict a single data point '''
            
            if tree.value!=None: return tree.value
            feature_val = x[tree.feature_index]
            if feature_val<=tree.threshold:
                return self.make_prediction(x, tree.left)
            else:
                return self.make_prediction(x, tree.right)
    ```
    
    ### Train-Test split
    
    ```python
    X = data.iloc[:, :-1].values
    Y = data.iloc[:, -1].values.reshape(-1,1)
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=41)
    ```
    
    ### Fit the model
    
    ```python
    classifier = DecisionTreeClassifier(min_samples_split=3, max_depth=3)
    classifier.fit(X_train,Y_train)
    classifier.print_tree()
    ```
    
    ### Test the model
    
    ```python
    Y_pred = classifier.predict(X_test) 
    from sklearn.metrics import accuracy_score
    accuracy_score(Y_test, Y_pred)
    ```
    

---

## Opracowanie filmu na podstawie naszego kodu  ⬇️

## Klasa Node:

```python
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

        def is_leaf_node(self):
            return self.value is not None
```

Klasa Node czyli węzeł dzieli się na dwa rodzaje: **decision node** oraz **leaf node**

**Decision Node:**

- Zawiera warunek definiowany przez `feature` oraz próg definiowany przez `threshold`
- Zawiera lewy `left` i prawy `right`, które prowadzą nas do dzieci tego węzła
- Zawiera information gain `info_gain` (którego tu nie mamy), wyznaczony przez split, aby ocenić wartość węzła

**Leaf Node:**

- Nie zawiera poprzednich wartości, a jedynie `value` ,  które jest “majority class of the leaf node”
- Pomaga nam zdefiniować klasę nowego data point, jeśli data point kończy w tym konkretnym liściu

## Klasa Tree

Co ona zawiera?

- W tej klasie jest zaimplementowany cały algorytm.
- Klasa zawiera metodę `budowania drzewa`
- Zawiera metodę obliczania `entropii`
- Zawiera metodę `split`
- Zawiera metodę `prediction`
- etc.

### Konstruktor

- **Implementacja tego gościa:**
    
    ```python
        def __init__(self, min_samples_split=2, max_depth=2):
            ''' constructor '''
            
            # initialize the root of the tree 
            self.root = None
            
            # stopping conditions
            self.min_samples_split = min_samples_split
            self.max_depth = max_depthA
    ```
    

W konstruktorze mamy zdefiniowane trzy atrybuty, które będą używane przez różne funkcje, które zdefiniujemy w klasie

```python
def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
	self.min_samples_split = min_samples_split
  self.max_depth = max_depth
  self.n_features = n_features
  self.root = None
```

- `Root`: potrzebujemy żeby trawersować drzewo
- `min_samples_split`, `max_depth`: warunki stopu. 
Jeśli w którymś z węzłów liczba próbek jest mniejsza od `min_samples_split`, to nie dzielimy dalej tego węzła i traktujemy go jako liść.
Jeśli głębokość drzewa osiągnie `max_depth`, również nie dzielimy dalej tego węzła.

### Budowanie drzewa

- **Implementacja tego gościa:**
    
    ```python
        def build_tree(self, dataset, curr_depth=0):
            ''' recursive function to build the tree ''' 
            
            X, Y = dataset[:,:-1], dataset[:,-1]
            num_samples, num_features = np.shape(X)
            
            # split until stopping conditions are met
            if num_samples>=self.min_samples_split and curr_depth<=self.max_depth:
                # find the best split
                best_split = self.get_best_split(dataset, num_samples, num_features)
                # check if information gain is positive
                if best_split["info_gain"]>0:
                    # recur left
                    left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                    # recur right
                    right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                    # return decision node
                    return Node(best_split["feature_index"], best_split["threshold"], 
                                left_subtree, right_subtree, best_split["info_gain"])
            
            # compute leaf node
            leaf_value = self.calculate_leaf_value(Y)
            # return leaf node
            return Node(value=leaf_value)
    ```
    

Najważniejsza funkcja tej klasy. Rekursywna funkcja budująca drzewo binarne przy użyciu rekursji.

```python
def _grow_tree(self, X, y, depth=0):
    pass
```

- First of all, dzieli `cechy features` i `cele targets` na dwie różne zmienne: `X` i `y`
- Wyodrębnia liczbę próbek i cech dzięki funkcji `np.shape()`
- W pierwszym ifie sprawdza wspomniane wyżej warunki stopu.
- Jeśli nie są spełniane, to dzielimy węzeł przy użyciu funkcji `_best_split` żeby uzyskać najlepszy podział.
- Kiedy mamy już najlepszy podział, sprawdzamy czy `information_gain` z tego best splita jest większy niż 0. Jeśli jest zero, to węzeł jest czysty i zawiera tylko jeden typ klasy.
- Jeśli tak, tworzymy `lewe poddrzewo` i `prawe poddrzewo`. Tutaj używamy rekursji, ponieważ wywołujemy tutaj funkcje budującą drzewo podczas gdy jesteśmy w tej funkcji. Najpierw stworzy się całe lewe poddrzewo i jak osiągnie liścia, to wtedy całe prawe poddrzewo.
Trzeba pamiętać, aby wywołując funkcje budującą poddrzewa zwiększyć zmienną `depth` o 1.
- Kiedy poddrzewa są gotowe, zwracamy węzeł. Jego atrybuty, czyli `feature`,  `threshold`, `left`, `right` (oraz `information_gain`, którego nie mamy w naszej implementacji) otrzymujemy z funkcji `_best_split` .  Ten węzeł będzie węzłem decyzyjnym.
- W implementacji gościa możemy zauważyć że `best_split` jest właściwie słownikiem, który jest zwracany przez funkcje.
- Kiedy mamy to wszystko, możemy obsłużyć sytuacje w której mamy węzeł liścia. W tym celu obliczamy wartość liścia `value` (koleś używa funkcji `calculate_leaf_value`. Po czym zwracamy ten węzeł liścia, podając jeden argument którym jest `value`.

### Podział węzła

- **Implementacja tego gościa:**
    
    ```python
        def get_best_split(self, dataset, num_samples, num_features):
            ''' function to find the best split '''
            
            # dictionary to store the best split
            best_split = {}
            max_info_gain = -float("inf")
            
            # loop over all the features
            for feature_index in range(num_features):
                feature_values = dataset[:, feature_index]
                possible_thresholds = np.unique(feature_values)
                # loop over all the feature values present in the data
                for threshold in possible_thresholds:
                    # get current split
                    dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                    # check if childs are not null
                    if len(dataset_left)>0 and len(dataset_right)>0:
                        y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                        # compute information gain
                        curr_info_gain = self.information_gain(y, left_y, right_y, "gini")
                        # update the best split if needed
                        if curr_info_gain>max_info_gain:
                            best_split["feature_index"] = feature_index
                            best_split["threshold"] = threshold
                            best_split["dataset_left"] = dataset_left
                            best_split["dataset_right"] = dataset_right
                            best_split["info_gain"] = curr_info_gain
                            max_info_gain = curr_info_gain
                            
            # return best split
            return best_split
    ```
    

Zwraca słownik, dzięki któremu jesteśmy w stanie utworzyć nowy węzeł.

```python
        def _best_split(self, X, y, feat_idxs):
            pass
```

- Tworzymy pusty słownik `best_split` oraz zmienną `max_info_gain`, która na początku ma wartość najmniejszej ujemnej liczby, ponieważ chcemy zmaksymalizować `information_gain` i do tego musimy użyć liczby, która jest mniejsza od jakiejkolwiek innej liczby.
- Robimy pętle przez wszystkie cechy `features` i w tej pętli musimy trawersować przez wszystkie możliwe wartości `threshold`.
- `features` to **liczby rzeczywiste**, a tych jest nieskończenie wiele. Nie ma zatem sensu żeby iterować przez wszystkie możliwe **liczby rzeczywiste**, bo będzie to **pętla nieskończona**, zatem będziemy iterować przez wszystkie możliwe **liczby rzeczywiste**, które zawierają się w naszym **dataset**. Pomaga nam w tym funkcja `np.unique()`, która zwraca unikalne wartości konkretnych cech. Dzięki temu możemy trawersować przez wszystkie możliwe wartości `features`.
- Teraz tworzymy drugą pętlę w środku. Tutaj na początku musimy podzielić **dataset** bazując na aktualnym `feature` i aktualnym `threshold`.  Używamy do tego funkcji `split`.
- Tworzy nam to `lewe i prawe poddrzewo`. Jeśli nie są one puste, to wyodrębniamy wartości docelowe, które są oznaczone przez `y`.  Robimy to również dla poddrzew (`left_y`, `right_y`).
- Następnie obliczamy `information_gain` przy użyciu funkcji `_information_gain()`. Można to zrobić dzięki `gini_index`.
- Jeśli mamy już `information_gain`, sprawdzamy, czy jest większy od `max_information_gain`. Jeśli tak, aktualizujemy nasz `best_split`.
- Kiedy przejedziemy przez wszystkie pętle zwracamy `best_split`.

### Funkcja split()

- **Implementacja tego gościa:**
    
    ```python
    def split(self, dataset, feature_index, threshold):
            ''' function to split the data '''
    
          dataset_left= np.array([rowfor rowin datasetif row[feature_index]<=threshold])
          dataset_right= np.array([rowfor rowin datasetif row[feature_index]>threshold])
    return dataset_left, dataset_right
    ```
    

Używamy jej w funkcji `_best_split()` aby robić nasze podziały. Funkcja ta przyjmuje argumenty `dataset`, `feature`, `threshold` i dzieli węzeł na dwie części. Pierwsza idzie do lewego dziecka, druga idzie do prawego dziecka.

- Używamy do tego list, do **lewej części** przypisujemy wszystkie wiersze, które są **większe, lub równe** od naszego `threshold`, do **prawej części** przypisujemy wiersze, które są **mniejsze**.
- Następnie zwracamy obie części.

```python
        def _split(self, X, y, split_thresh):
            pass
```

### Funkcja information_gain()

- **Implementacja tego gościa:**
    
    ```python
    def information_gain(self, parent, l_child, r_child, mode="entropy"):
            ''' function to compute information gain '''
            
            weight_l = len(l_child) / len(parent)
            weight_r = len(r_child) / len(parent)
            if mode=="gini":
                gain = self.gini_index(parent) - (weight_l*self.gini_index(l_child) + weight_r*self.gini_index(r_child))
            else:
                gain = self.entropy(parent) - (weight_l*self.entropy(l_child) + weight_r*self.entropy(r_child))
            return gain
    ```
    

Ta funkcja oblicza nam i zwraca `information_gain()`. Odejmuje sumę `information_gain` dzieci od węzła głównego. Koleś używa tutaj zmiennych `weigth_l` i `weight_r` żeby przechowywać rozmiary dzieci węzła podzielone przez rozmiar węzła. Następnie używamy wzoru z entropią na wyliczenie `information_gain` i je zwracamy.

### Funkcja entropy()

- **Implementacja tego gościa:**
    
    ```python
    def entropy(self, y):
            ''' function to compute entropy '''
            
            class_labels = np.unique(y)
            entropy = 0
            for cls in class_labels:
                p_cls = len(y[y == cls]) / len(y)
                entropy += -p_cls * np.log2(p_cls)
            return entropy
    ```
    

Oblicza entropię ze wzoru. Jest zdefiniowana już i nie musimy jej pisać.

### Funkcja gini_index()

- **Implementacja tego gościa:**
    
    ```python
    def gini_index(self, y):
            ''' function to compute gini index '''
            
            class_labels = np.unique(y)
            gini = 0
            for cls in class_labels:
                p_cls = len(y[y == cls]) / len(y)
                gini += p_cls**2
            return 1 - gini
    ```
    

Oblicza index Gini, który pozwala nam również na obliczenie `information_gain` i nie trzeba wtedy entropi. My używamy entropi więc to nam zbędne.

### Funkcja calculate_leaf_value()

- **Implementacja tego gościa:**
    
    ```python
    def calculate_leaf_value(self, Y):
            ''' function to compute leaf node '''
            
            Y = list(Y)
            return max(Y, key=Y.count)
    ```
    

Oblicza `value` liścia, czyli główną klasę występującą w tym konkretnym liściu. Musimy więc znaleźć najczęściej występujacy element w `y`.

### Funkcja print_tree()

- **Implementacja tego gościa:**
    
    ```python
    def print_tree(self, tree=None, indent=" "):
            ''' function to print the tree '''
            
            if not tree:
                tree = self.root
    
            if tree.value is not None:
                print(tree.value)
    
            else:
                print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
                print("%sleft:" % (indent), end="")
                self.print_tree(tree.left, indent + indent)
                print("%sright:" % (indent), end="")
                self.print_tree(tree.right, indent + indent)
    ```
    

My tego nie implementujemy. Wyświetla drzewo w sposób rekursywny przez trawersowanie pre-order.

### Funkcja fit()

- **Implementacja tego gościa:**
    
    ```python
    def fit(self, X, Y):
            ''' function to train the tree '''
            
            dataset = np.concatenate((X, Y), axis=1)
            self.root = self.build_tree(dataset)
    ```
    

Na początku konkatenujemy (łączymy) `x` i `y`, aby stworzyć nasz dataset, a następnie wywołujemy funkcje `grow_tree()`, która zwróci nam `root node`, a ten zapisujemy do naszego `self.root`.

### Funkcja predict()

- **Implementacja tego gościa:**
    
    ```python
    def predict(self, X):
            ''' function to predict new dataset '''
            
            preditions = [self.make_prediction(x, self.root) for x in X]
            return preditions
    ```
    

Bierze jakiś **dataset** i zwraca nam odpowiednie `predictions`. W tej funkcji wywołujemy funkcje `make_prediction()` dla całej tablicy. Następnie zwracamy `predictions`.

### Funkcja make_prediction()

- **Implementacja tego gościa:**
    
    ```python
    def make_prediction(self, x, tree):
            ''' function to predict a single data point '''
            
            if tree.value!=None: return tree.value
            feature_val = x[tree.feature_index]
            if feature_val<=tree.threshold:
                return self.make_prediction(x, tree.left)
            else:
                return self.make_prediction(x, tree.right)
    ```
    

Funkcja ta u nas nazywa się `_traverse_tree()` i pobiera ona w argumencie `node`. 

- Początkowo sprawdzamy w funkcji czy `node.value` nie jest `None`, zwracamy `node.value`. Oznacza to, że węzeł jest liściem.
- Jeśli ten warunek jest spełniony, czyli jest `None`, wtedy jest to węzeł decyzyjny, zatem przypisujemy do zmiennej wartość `x` spod indeksu `node.feature`.
- Następnie sprawdzamy, czy ta zmienna ma wartość mniejszą bądź równą `node.threshold`.
- **Jeśli tak**, wywołujemy rekursywnie funkcję, z tym samym parametrem `x`, natomiast jako węzeł podajemy `node.left`.
- **Jeśli nie**, wywołujemy rekursywnie funkcję z tym samym parametrem `x` i jako węzeł podajemy `node.right`.