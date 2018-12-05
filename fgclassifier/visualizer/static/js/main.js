/**
 * Fine-grain Sentiment Analysis Visualizer
 * 
 * (c)2018  Jianchao Yang
 */
import SingleReviewChart from './single_review.js'
import ModelStatsUpdater from './model_stats.js'

document.addEventListener("DOMContentLoaded", function(event) {
  // init and render
  let single = new SingleReviewChart('#single-review', 'single');
  let model_stats = new ModelStatsUpdater('#single-review', 'global');
  let opts = {
    qs: location.search,
    updateHistory: false
  }
  single.fetchAndUpdate(opts);
  model_stats.fetchAndUpdate(opts);

  window.single = single
  window.model_stats = model_stats
})
