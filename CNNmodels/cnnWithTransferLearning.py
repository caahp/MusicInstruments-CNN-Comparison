# Importando bibliotecas necessárias
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array

# Função para aplicar estiramento de contraste
def contrast_stretching(image):
    xp = [0, 64, 128, 192, 255]
    fp = [0, 16, 128, 240, 255]
    stretched_image = np.interp(image, xp, fp).astype(np.uint8)
    return stretched_image

# Função para aplicar filtros de contorno e correção de cor
def add_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = gray.astype(np.uint8)
    gray = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    image = image.astype(np.float32)
    edges_colored = edges_colored.astype(np.float32)
    combined = cv2.addWeighted(image, 0.8, edges_colored, 0.2, 0)
    combined = combined.astype(np.uint8)
    return combined

# Função de pré-processamento personalizada
def custom_preprocessing(image):
    image = cv2.resize(image, (150, 150))
    image = contrast_stretching(image)
    image = add_edge_detection(image)
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
    validation_split=0.2  # 20% dos dados serão usados para validação
)

# Gerador para o conjunto de teste sem aumento de dados
test_datagen = ImageDataGenerator(preprocessing_function=custom_preprocessing)

# Diretórios dos dados de treino e teste
data_dir = '/kaggle/input/instrumentsdataset/train'
test_dir = '/kaggle/input/instrumentsdataset/validation'

# Gerador de treino
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'  # Subconjunto de treino
)

# Gerador de validação
val_generator = train_datagen.flow_from_directory(
    data_dir,
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
    shuffle=False  # Não misturar os dados para garantir correspondência nas métricas
)

# Carregando o modelo InceptionV3 pré-treinado, sem as camadas de saída
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Congelando as camadas do modelo base para não treinar novamente
base_model.trainable = False

# Adicionando novas camadas ao modelo
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

# Criando o modelo completo
model = Model(inputs=base_model.input, outputs=predictions)

# Compilando o modelo com uma taxa de aprendizado ajustada
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Treinando o modelo com Transfer Learning
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=200,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size
)

# Obtendo os dados do histórico de treino
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

# Plotando gráfico de loss para treino e validação
plt.figure(figsize=(10, 6))
plt.plot(epochs, loss, 'b', label='Train Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

plt.figure(figsize=(10, 6))
plt.plot(epochs, accuracy, 'b', label='Train Accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Avaliação final no conjunto de teste
test_loss, test_acc = model.evaluate(test_generator)
print(f'Acurácia no conjunto de teste: {test_acc*100:.2f}%')

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