# Importando bibliotecas necessárias
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

# Função para aplicar estiramento de contraste
def contrast_stretching(image):
    # Normaliza a imagem para valores entre 0 e 255
    xp = [0, 64, 128, 192, 255]
    fp = [0, 16, 128, 240, 255]
    stretched_image = np.interp(image, xp, fp).astype(np.uint8)
    return stretched_image

# Função para aplicar filtros de contorno e correção de cor
def add_edge_detection(image):
    # Converte a imagem para escala de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Certifica-se de que a imagem em escala de cinza é do tipo correto
    gray = gray.astype(np.uint8)
    # Equaliza o histograma da imagem para melhorar contraste
    gray = cv2.equalizeHist(gray)
    # Aplicando filtro gaussiano para suavizar a imagem
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Aplicando detecção de bordas (Canny)
    edges = cv2.Canny(blurred, 100, 200)
    # Convertendo as bordas para 3 canais
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    # Garantindo que ambas as imagens tenham o mesmo tipo e profundidade
    image = image.astype(np.float32)
    edges_colored = edges_colored.astype(np.float32)
    # Combina a imagem original com as bordas para destacar formas
    combined = cv2.addWeighted(image, 0.8, edges_colored, 0.2, 0)
    # Converte a imagem combinada de volta para o formato de imagem padrão
    combined = combined.astype(np.uint8)
    return combined

# Função de pré-processamento personalizada
def custom_preprocessing(image):
    # Redimensionando a imagem
    image = cv2.resize(image, (150, 150))
    # Aplicando estiramento de contraste
    image = contrast_stretching(image)
    # Aplicando detecção de bordas e correção de cor
    image = add_edge_detection(image)
    # Normalizando a imagem
    image = img_to_array(image).astype('float32') / 255.0
    return image


# Configuração dos geradores de imagem com aumento de dados
train_datagen = ImageDataGenerator(
    preprocessing_function=custom_preprocessing,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.4,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest',
    validation_split=0.2  # Agora o conjunto de validação vem dos dados de treino
)

# Gerador de teste sem aumento de dados
test_datagen = ImageDataGenerator(preprocessing_function=custom_preprocessing)

# Diretórios dos dados de treino e teste
train_dir = '/kaggle/input/instrumentsdataset/train'
test_dir = '/kaggle/input/instrumentsdataset/validation'  # Agora a pasta validation será usada para teste

# Geradores de treino e validação a partir da pasta train
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'  # Subconjunto de treino
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'  # Subconjunto de validação
)

# Gerador de teste
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Para garantir a correspondência entre previsões e rótulos verdadeiros
)

# Definindo o modelo de rede neural
model = Sequential([
    Input(shape=(150, 150, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(512, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(train_generator.num_classes, activation='softmax')
])

# Compilando o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Definindo callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=250, restore_best_weights=True)

# Treinando o modelo com os dados processados
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=600,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    callbacks=[early_stopping]
)

# Salvando o modelo treinado
model.save('modelo_instrumentos_melhorado.h5')

# Avaliação final com o conjunto de teste
test_loss, test_acc = model.evaluate(test_generator)
print(f'Acurácia no conjunto de teste: {test_acc * 100:.2f}%')

# Gerando previsões no conjunto de teste
test_generator.reset()
predictions = model.predict(test_generator, steps=test_generator.samples // test_generator.batch_size + 1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Plotando a matriz de confusão
conf_matrix = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predição')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()

# Relatório de classificação
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

# Cálculo das métricas
accuracy = accuracy_score(true_classes, predicted_classes)
precision = precision_score(true_classes, predicted_classes, average='weighted')
recall = recall_score(true_classes, predicted_classes, average='weighted')
f1 = f1_score(true_classes, predicted_classes, average='weighted')

# Exibindo as métricas
print(f'Acurácia: {accuracy:.2f}')
print(f'Precisão: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
