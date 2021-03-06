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

const scoreColor = (d, mu, sd) => {
  if (!d) return 'gray';
  if (d < mu - sd) {
    return '#f46d43';
  }
  return '#666';
}

export default class ModelStatsUpdater extends SingleReviewChart {

  constructor(elem, name) {
    super(elem, name)
    this.fields = ['dataset', 'fm', 'clf']
    this.endpoint = '/model_stats'
    this.loaderDelay = 300
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
    if (typeof avg_score == 'undefined') {
      avg_score = '-'
    } else {
      avg_score = avg_score.toFixed(3)
    }
    let all_scores = this.data['scores'] || Array(20)
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
        .data(all_scores).enter()
        .append('text')
        .attr('class', 'cat-score')
        .attr('text-anchor', 'middle')
        .attr('transform', (d, i) =>
          `translate(20, ${barHeight / 2 + 5 + i * (barHeight + 1)})`)
    } else {
      title = title.transition().duration(300)
      g = g.transition().duration(300)
      scores = scores.data(all_scores)
        .transition().duration(300)
    }
    title.attr('transform', `translate(${xoffset - 10 + scoreWidth / 2}, 13)`)
    g.attr('transform', `translate(${xoffset}, 20)`)

    let mu = avg_score
    let sd = d3.deviation(all_scores)
    scores.text((d) => d ? d.toFixed(3) : '-')
      .style('fill', (d) => scoreColor(d, mu, sd))

    this.html('.overall-f1-score', avg_score)
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
    let true_dist, has_true_dist
    if (this.data['true_dist']) {
      has_true_dist = true
      true_dist = this.data['true_dist']
    } else {
      has_true_dist = false
      true_dist = d3.range(20).map((i) => [1, 0, 0, 0]);
    }
    this.buildBricks(true_dist, 'global actual', {
      barheight,
      barmargin,
      yoffset: yoffset + barheight,
      xoffset: xoffset,
      showtitle: false,
      fullwidth: width,
      alpha: has_true_dist ? 0.7 : 0.5,
      tooltipTmpl: (d, i) => {
        if (!has_true_dist) {
          return ''
        }
        let dd = d[1] - d[0]
        if (dd < 1) {
          dd = dd.toFixed(2);
        }
        return `Actual<br>${labeltext[i]}: ${dd}`
      }
    })

  }

}
