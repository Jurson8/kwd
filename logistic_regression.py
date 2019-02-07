import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

# wczytanie danych
patients = datasets.load_breast_cancer()

#domyślny podział
patients_train_data, patients_test_data, \
patients_train_target, patients_test_target = \
train_test_split(patients.data,patients.target, test_size=0.1)

mali = 0
ben = 0
for target in patients.target:
    if ( target == 1) :
        ben = ben + 1
    else :
        mali = mali + 1


print(str(mali) + ' ' + str(ben) + ' ' + str(patients.target.size))

#prezentacja danych, opis, wizualizacja
print(patients.DESCR)
plt.bar(['Złośliwe', 'Łagodne'], [mali, ben])
plt.show()

# standaryzacja, ale tutaj nie jest konieczna
# patients_train_data = preprocessing.StandardScaler().fit_transform(patients_train_data)
# patients_test_data = preprocessing.StandardScaler().fit_transform(patients_test_data)

logistic_regression = LogisticRegression()
logistic_regression.fit(patients_train_data, patients_train_target)

acc1 = accuracy_score(patients_test_target, logistic_regression.predict(patients_test_data))
print("Dokładność modelu z domyślnym podziałem to {0:0.2f}".format(acc1))

# przewidywanie dla konkretnego przypadku
# id=6
# prediction = logistic_regression.predict(patients_test_data[id,:].reshape(1,-1))
# print("Prognoza modelu dla pacjenta o id {0} to wartość {1}".format(id, prediction))
#
# print("Rzeczywista wartość dla pacjenta o id \"{0}\" to {1}".format(id, patients_test_target[id]))



# cross validacja - k-krotna walidacja
# na ile podzbiorów dzielimy dane do cross validacji
cv_size = 7

scores = cross_val_score(logistic_regression, patients.data, patients.target, cv=cv_size)
print(scores)
best_set_index = np.argmax(scores)
print(np.argmax(scores))
print(patients.data.shape[0])

test_start_index = int(patients.data.shape[0]/cv_size) * best_set_index
test_end_index = int(patients.data.shape[0]/cv_size) + test_start_index


# wybranie najlepszych danych dla modelu
patients_test_data = patients.data[test_start_index : test_end_index]
patients_train_data = np.delete(patients.data, range(test_start_index,test_end_index), 0)
patients_test_target = patients.target[test_start_index : test_end_index]
patients_train_target = np.delete(patients.target, range(test_start_index,test_end_index), 0)

logistic_regression.fit(patients_train_data, patients_train_target)

# prognoza dla konkretnego pacjenta
id=15
prediction = logistic_regression.predict(patients_test_data[id,:].reshape(1,-1))
print("Prognoza modelu dla pacjenta o id {0} to wartość {1}".format(id, prediction))

print("Rzeczywista wartość dla pacjenta o id \"{0}\" to {1}".format(id, patients_test_target[id]))

# trafność dla danego id
prediction_probability = logistic_regression.predict_proba(patients_test_data[id,:].reshape(1,-1))
print(prediction_probability)

# jakość i trafność całego modelu
acc2 = accuracy_score(patients_test_target, logistic_regression.predict(patients_test_data))
print("Dokładność modelu z cross validacją to {0:0.2f}".format(acc2))

conf_matrix = confusion_matrix(patients_test_target, logistic_regression.predict(patients_test_data))

# macierz konfuzji - sprawdzenie jakości
print(conf_matrix)
print("Dzięki cross validacji skuteczność modelu udało się poprawić o {0:0.2f}%.".format(acc2*100 - acc1*100))






