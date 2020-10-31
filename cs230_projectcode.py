import requests
from PIL import Image
from io import BytesIO
import pickle
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import UnidentifiedImageError
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
import random
import math

#use the TMDB API to gather urls that access pictures of movie posters and movie background pictures
def gather_image_urls():
  #a csv from a kaggle dataset that has links to the poster pics but not background pictures so need the API
  data = pandas.read_csv('/movies_metadata.csv') 
  movies = data['id']
  misses = 0
  genres_back = {}
  #possibly use later for gathering images by director and other metadata
  id_json = {}

  #gather all images (poster and background) and hash by genre name (EDIT decided to use only background pics)
  for i in range(movies.size):
    if i % 100 == 0:
      print(i)
    r = requests.get(f'https://api.themoviedb.org/3/movie/{movies[i]}?api_key=82ca501ce9ffac46062e9c3587b3c92b&language=en-US')
    r = r.json()
    id_json[i] = r
    
    #organize the paths to the urls by genre (they only have one)
    if 'backdrop_path' in r and r['backdrop_path'] is not None:
      back_path = r['backdrop_path']
      for genre in r['genres']:
        if genre['name'] in genres_back:
          genres_back[genre['name']].append(back_path)
        else: genres_back[genre['name']] = [back_path]  
    else:
      misses += 1
  return (genres_back,id_json)

#we store the urls to the pics as .pickle files for ease
def retrieve_url_data():
  with open('genres_back.pickle','rb') as f:
    genres_back = pickle.load(f)

  with open('id_json.pickle', 'rb') as f:
    id_json = pickle.load(f)

  return (genres_back, id_json)

#accessing the API to actually download the background pictures
def get_train_data(genres_back, key):
  X_back = []
  Y_back = []
  issues_id = []

  count = 0
  for img in genres_back[key]:
    img = requests.get(f'https://image.tmdb.org/t/p/w500{img}')
    try:
      img = Image.open(BytesIO(img.content)) 
    except UnidentifiedImageError:
      print('problem')
      issues_id.append(count)
      continue
    img = img.resize((400, 400)) 
    #convert the images into arrays
    img = image.img_to_array(img)
    img = img/255
    X_back.append(img)
    Y_back.append(key)
    count+=1
    if count % 500 == 0:
      print(count)
    if count == 2500:
      break

  with open(f"X_back_{key}.pickle", 'wb') as f:
    pickle.dump(X_back, f)

  with open(f"Y_back_{key}.pickle", 'wb') as f:
    pickle.dump(Y_back, f)

  return (X_back, Y_back, issues_id)

#function to access the data by genre. We do this because the files are multiple GBs
def retrieve_images(image_set):
  
  with open(f'{image_set}_Animation.pickle','rb') as f:
    Animation = pickle.load(f)
    print(1)

  with open(f'{image_set}_Fantasy.pickle', 'rb') as f:
    Fantasy = pickle.load(f)
    print(2)

  with open(f'{image_set}_Horror.pickle', 'rb') as f:
    Horror = pickle.load(f)
    print(3)

  with open(f'{image_set}_Science_Fiction.pickle', 'rb') as f:
    ScienceFiction = pickle.load(f)
    print(4)

  with open(f'{image_set}_War.pickle','rb') as f:
    War = pickle.load(f)
    print(5)

  with open(f'{image_set}_Western.pickle', 'rb') as f:
    Western = pickle.load(f)
    print(6)

  return (Animation, Fantasy, Horror, 
  ScienceFiction, War, Western)

#forgot to initially turn the image arrays into numpy, hence this function
def numpyify():
  genre_names = ['X_back_Animation','X_back_Fantasy','X_back_Crime','X_back_Horror', 'X_back_Romance', 
  'X_back_Science_Fiction', 'X_back_War', 'X_back_Western']
  count = 0
  for genre in all_X[3:]:
    trains = []
    for i in range(len(genre)):
      if np.array(genre[i]).shape == (400,400,3):
        trains.append(genre[i])
    genre = trains
    genre = np.array(genre)

    with open(f"{genre_names[count]}.pickle", 'wb') as f:
      pickle.dump(genre, f, protocol=4)
    print(genre_names[count])

    count +=1

#function for combining all the specific genre files
def combine_train_data():
  X_images = retrieve_images('X_back')

  Xs = np.concatenate((X_images[0], X_images[1], X_images[2], X_images[3], X_images[4], X_images[5]),axis=0)
  print(Xs.shape)
  with open(f"X_All.pickle", 'wb') as f:
      pickle.dump(Xs, f, protocol=4)

#convert the Y labels to one hot
def one_hot():
  Y_images = retrieve_images('Y_back')
  Ys = np.concatenate((Y_images[3], Y_images[5]),axis=0)
  df = pd.DataFrame()
  df['Genre'] = Ys
  le = LabelEncoder()

  df = df.apply(le.fit_transform)

  enc = OneHotEncoder()

  enc.fit(df)

  onehotlabels = enc.transform(df).toarray()
  print(onehotlabels)

  with open(f"Y_All_small.pickle", 'wb') as f:
      pickle.dump(onehotlabels, f, protocol=4)

#method for resizing the image. (Wouldn't hard code it but all the models we used were 224,224)
def resize_image():
  trains = []
  for i in range(len(Xs)):
    im = Image.fromarray(Xs[i],"RGB")
    im = im.resize((224,224))
    im = image.img_to_array(im)
    trains.append(im)
  trains = np.array(trains)
  Xs = trains
  print(Xs.shape)

  with open(f"X_back_Western.pickle", 'wb') as f:
      pickle.dump(Xs, f, protocol=4)

#all purpose method for building, training, and saving the model. Settled on this model after some iteration
def train():
  with open(f'Y_All_small.pickle', 'rb') as f:
      Labels = pickle.load(f)
      print(Labels)

  with open(f'X_All_small.pickle', 'rb') as f:
      Xs = pickle.load(f)
      print(Xs.shape)

  X_train, X_test, y_train, y_test = train_test_split(Xs, Labels, test_size=0.1)
  print('have training split')

  basemodel = tf.keras.applications.VGG19(
      include_top=False,
      weights='imagenet',
      input_tensor=None,
      input_shape=(224,224,3),
      pooling=None,
  )
  model = tf.keras.Sequential(basemodel)
  model.add(Flatten())

  model.add(Dense(1024, activation = 'relu')) 
  model.add(Dense(128, activation = 'relu')) 
  model.add(Dense(6, activation = 'softmax'))

  model.summary()

  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=64)

  model.save('vgg19_model_transfer.h5')   #add in animation if you want'''

#used for prediciton and organizing the predictions for future use in gathering needed images
def predictions():
  with open(f'X_All_small.pickle', 'rb') as f:
      X_All = pickle.load(f)
      print(Xs.shape)

  predictions = model.predict(X_All)
  preds_by_genre = []
  X_Genres = retrieve_images('X_back')
  index = 0
  for i in range(len(X_Genres)):
    #will help to have the prediction split by genre
    preds_by_genre.append(predictions[index:index+len(X_Genres[i])])
    index += len(X_Genres[i])
  #(0:Animation,1:Fantasy,2:Horror,3:SciFi,4:War,5:Western)
  results = {'0':[],'1':[],'2':[],'3':[],'4':[],'5':[]}
  #use to keep track of index in X_All
  tracker = 0
  for genre in range(len(preds_by_genre)):
    #keep track of the indices of the images with the highest prob of each genre
    genre_preds = preds_by_genre[genre]
    top5 = [0,0,0,0,0]
    top5_ind = [0,0,0,0,0]
    for movie in range(len(genre_preds)):
      prob = genre_preds[movie][genre]
      for i in range(len(top5)):
        if prob > top5[i]:
          top5_ind[i] = movie + tracker
          top5[i] = prob
    results[genre] = top5_ind
    tracker += len(preds_by_genre)

#used for gathering and saving the images we want for future use in neural style transfer
def get_top_movies(results):
  with open('X_All.pickle','rb') as f:
    X_All = pickle.load(f)

  with open('X_embeddings', 'rb') as f:
    X_embeddings = pickle.load(f)

  top_pics[]
  #follow naming pattern, 5 top prob images + 5 random images of each genre
  random = [random.randint(0,2000) for i in range(5)]
  for i in results.keys():
    for index in range(len(i)):
      top_pic.append(i[index])
      (Image.fromarray(X_All[i[index]])).save(f'top_pic{i}.jpg')
      (Image.fromarray(X_All[random])).save(f'random_genre{i}.jpg')

  #5 closest to optimal image by Euclidean distance, same idea as above
  for i in top_pics:
    top_emb = X_embeddings[i]
    closest = [float(inf),float(inf),float(inf),float(inf)]
    closest_ind = [0,0,0,0]
    for emb in range(len(X_embeddings)):
      dist = np.linalg.norm(top_emb - X_embeddings[emb])
      for i in range(len(closest_ind)):
        if dist > closest_ind[i]:
          closest_ind[i] = emb
          closest[i] = dist
    for index in range(len(closest)):
      (Image.fromarray(X_All[index])).save(f'closest_pic{i}.jpg')
    (Image.fromarray(X_All[i])).save(f'top_pic_genre{i}.jpg')


    




