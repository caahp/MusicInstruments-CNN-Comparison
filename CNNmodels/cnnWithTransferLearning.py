import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Carregando o modelo InceptionV3 pré-treinado, sem as camadas de saída
# Testamos com ResNet50, MobileNetV2, VGG16 e InceptionV3
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Congelando as camadas do modelo base para não treinar novamente
base_model.trainable = False

# Adicionando novas camadas ao modelo
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(7, activation='softmax')(x)

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

# Avaliação final
loss, acc = model.evaluate(val_generator)
print(f'Acurácia: {acc*100:.2f}%')

# Gerando previsões no conjunto de validação
val_generator.reset()
predictions = model.predict(val_generator, steps=val_generator.samples // val_generator.batch_size + 1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = val_generator.classes
class_labels = list(val_generator.class_indices.keys())

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