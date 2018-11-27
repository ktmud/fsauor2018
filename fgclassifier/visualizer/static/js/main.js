/**
 * Fine-grain Sentiment Analysis Visualizer
 * 
 * (c)2018  Jianchao Yang
 */
import SingleReviewChart from './single_review.js'

document.addEventListener("DOMContentLoaded", function(event) {
  let single = new SingleReviewChart('#single-review');
  // init and render
  single.fetchAndUpdate(location.search, false);
  window.single = single
})
