import {
  labelWidth,
  statsWidth,
  barHeight,
  label2idx,
  labeltext
} from './consts.js'
import SingleReviewChart from './single_review.js'


// width for displaying the scores
let scoreWidth = 60

export default class ModelStatsUpdater extends SingleReviewChart {

  constructor(elem, name) {
    super(elem, name)
    this.fields = ['dataset', 'fm', 'clf']
    this.endpoint = '/model_stats'
  }

  render(rawData) {
    if (rawData) {
      this.data = rawData
    }
    if (this.data) {
      this.updateScores()
      this.updateDistBars()
    }
  }

  prepareMore() {

  }

  /**
   * Get X offset of the stats section
   */
  getXoffset() {
    return labelWidth + this.getFullwidth() + barHeight * 2 + 19
  }

  updateScores() {
    let xoffset = this.getXoffset() + statsWidth - scoreWidth + 10
    let avg_score = this.data['avg_score']
    let title = this.bricks.select('text.cat-score-title')
    let scores = this.bricks.selectAll('text.cat-score')
    let g = this.bricks.select('g.cat-scores')
    if (scores.size() == 0) {
      // add title text
      title = this.bricks.append('text')
        .attr('class', 'cat-score-title')
        .text('F1')
        .attr('text-anchor', 'middle')
      g = scores = this.bricks.append('g')
        .attr('class', 'cat-scores')
      scores = g.selectAll('g.scores')
        .data(this.data['scores']).enter()
        .append('text')
        .attr('class', 'cat-score')
        .attr('text-anchor', 'begin')
        .attr('transform', (d, i) =>
          `translate(0, ${barHeight / 2 + 5 + i * (barHeight + 1)})`)
    } else {
      title = title.transition().duration(300)
      g = g.transition().duration(300)
      scores = scores.data(this.data['scores'])
        .transition().duration(300)
    }
    title.attr('transform', `translate(${xoffset - 10 + scoreWidth / 2}, 13)`)
    g.attr('transform', `translate(${xoffset}, 20)`)
    scores.text((d) => d.toFixed(3))
      .style('fill', (d) => d < avg_score ? '#f46d43' : '#66bd63')
    
    this.html('.overall-f1-score', this.data['avg_score'].toFixed(3))
  }

  updateDistBars() {
    let xoffset = this.getXoffset()
    let width = statsWidth - scoreWidth
    let barheight = 14.5
    let barmargin = 15.5
    let yoffset = 20

    if (this.data['predict_dist']) {
      this.buildBricks(this.data['predict_dist'], 'global', {
        barheight,
        barmargin,
        yoffset,
        xoffset: xoffset,
        titlex: width / 2,
        titley: -8,
        fullwidth: width,
        alpha: 0.8,
        tooltipTmpl: (d, i) => {
          let dd = d[1] - d[0]
          if (dd < 1) {
            dd = dd.toFixed(2);
          }
          return `Predicted<br>${labeltext[i]}: ${dd}`
        }
      })
    }
    // update true distribution
    if (this.data['true_dist']) {
      this.buildBricks(this.data['true_dist'], 'global actual', {
        barheight,
        barmargin,
        yoffset: yoffset + barheight,
        xoffset: xoffset,
        showtitle: false,
        fullwidth: width,
        alpha: 0.7,
        tooltipTmpl: (d, i) => {
          let dd = d[1] - d[0]
          if (dd < 1) {
            dd = dd.toFixed(2);
          }
          return `Actual<br>${labeltext[i]}: ${dd}`
        }
      })
    }

  }

}
