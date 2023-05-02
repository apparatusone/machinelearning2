import tensorflow as tf
import tensorflow_text as text

dataset_name = 'imdb'
saved_model_path = './{}_bert_small'.format(dataset_name.replace('/', '_'))

reloaded_model = tf.saved_model.load(saved_model_path)

def print_my_examples(inputs, results):
  result_for_printing = \
    [f'input: {inputs[i]:<30} : score: {results[i][0]:.6f}'
                         for i in range(len(inputs))]
  print(*result_for_printing, sep='\n')
  print()


examples = [
    "I loved every minute of it",
    "As a purely fluff piece - it's fun, bright spirited, and enjoyable for families. A movie that has all of the elements you'd want represented from Super Mario bros video games, and as a bonus, it also showcases Mario cart and donkey kong junior. The animation is strong, the worlds are exactly like the video games and truly fun moments for kids. The runtime is great and the movie moves along without pause. AND I expect better, higher quality movie making. The voice acting in this movie is so lazy. Chris Pratt is forgettable-anyone could have done better voicing such a beloved character. Princess Peach is the same - voice acting is boring and fairly emotionless. The worst voice acting belongs to Seth's terrible Donkey Kong junior - not a single attempt to make an actual character voice, just Seth's normal voice slapped on Donkey Kong. Toad's voice acting is also meatless and without any real jokes. The script is mostly empty and there is little to no attempt at character motivation, organic character decision making - for example, why Princess Peach tells Mario he can come with her to save her kingdom. Almost every opportunity to make a joke is skipped. None of the characters have stated goals aside from King Koppa. And saddest of all to me, there is no attempt to create backstory or give context to these beloved worlds and characters."    'The movie was great!',
    'The movie was meh.',
    'The movie was okish.',
    'The movie was terrible...'
]

reloaded_results = tf.sigmoid(reloaded_model(tf.constant(examples)))

print('Results from the saved model:')
print_my_examples(examples, reloaded_results)