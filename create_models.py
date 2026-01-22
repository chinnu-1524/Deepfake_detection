from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense

# Create model
model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(224,224,3)),
    Flatten(),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy')

# Save models
model.save("model/image_model.h5")
model.save("model/video_model.h5")

print("image_model.h5 and video_model.h5 saved successfully!")
