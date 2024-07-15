<h1> Genre Scoring Model (Natural Language Processing Project) </h1>
        <p><strong>Overview:</strong> This is a NLP project, where we try to score the genre's for a movie, using the summary of the movie. <br>
        The list of genre's that will be scored are:<br>
        ['Action', 'Adult', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'Film-Noir', 'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western']</p>
<h5> Approach for the Project</h5>
<p> For this project, we used two different approaches. </p>
<OL>
  <li> <b> The Neural Network Approach </b> A neural network model was trained on the data, where the neural network was build with the help of PyTorch. The embeddings of the summary was created using Sentence BERT (SBERT) and then it was passed into the neural network. The neural network was a simple ANN. 'Ranger' optimizer was used for optimization purposes, which incorporates several improvements over Adam, such as gradient centralization and adaptive momentum estimation. This helped in making the training of the model a bit more faster. We initally trained this model as a MutliLabelClassifier, that classified the movies into different labels (genres in these case) and then we just used the probability score as the genre scores.</li>
  <li> <b> The Cosine Similarity Approach </b> First we created a detailed description of each genre (using ChatGPT) and then we simply used cosine similarity to see, which movie summary embedding and genre description embedding are similar to each other, and according scored each genre. Again we used Sentence BERT (SBERT) for creating these embeddings. </li>
</OL>
<p> In both the cases, the scores were scaled up, between 0 to 10, using Min-Max Scaler, which was coded from scratch rather than using the sklearn library.</p>


<h6> The predictios made by both the approaches has been attached, and it was obsereved on different test datasets, that the Neural Network Approach seemed to perform better than the Cosine Similarity Approach. Not all the test predictions has been uploaded, only the ones in which each apporach was seen to be the most accurate, has been uploaded.</h6>


<h2> Conclusion drawn: </h2>
<p> The model, predicts a large part of the genre's right, hence scoring them accurately. But then there are a lot of things that need to be kept in mind:</p>
<ol>
  <li> The summary needs to be a proper summary of the movie. That means, it must not be a one liner, or must not tell something in general about the movie and must definitely not be ambiguous. The summary must cover the major plot of the movie, in order for the correct scoring of the genres.</li>
  <li> The model's prediction might not be 100% accurate, as it just provides the genre scores, based on the summary. The model does not know, if the given event is a real event that took place, if the movie is an animated movie or not, the movie is a short of not (unless specified in the summary), hence the prediction cannot be considered 100% accurate.</li>
  <li> The predictions of ChatGPT for the exact same movies has been also uploaded. There is a clear difference seen in the scoring between the two models. The reason I believe that is so, is because, ChatGPT is trained on tons of data, making it more well versed and knowledgeable compared to our model, hence if our model also gets more data, then its performance can definitly be increased.</li>
</ol>
