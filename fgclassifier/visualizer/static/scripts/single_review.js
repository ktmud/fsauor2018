/**
 * Single Review predictions
 */
import { $ } from "./helpers.js"
import AsyncUpdater from './async_updater.js'

class SingleReviewChart extends AsyncUpdater {

  constructor(elem) {
    super(elem)
    // to make a prediction, we need all parameters
    this.fields = ['dataset', 'keyword', 'seed', 'fm', 'clf']
    this.endpoint = '/predict'

    this.root.select('.prev-seed').on('click', () => {
      let elem = $('#inp-seed')
      elem.value = Math.max(parseInt(elem.value, 0) - 1, 0)
      this.fetchAndUpdate()
    })
    this.root.select('.next-seed').on('click', () => {
      let elem = $('#inp-seed')
      elem.value = Math.min(parseInt(elem.value, 0) + 1, 99999999)
      this.fetchAndUpdate()
    })
    this.root.selectAll('select, input').on('change', () => {
      this.fetchAndUpdate()
    })
  }

  prepare(elem) {
    super.prepare(elem)
    this.bricks = this.root.select('.bricks')
    this.bars = this.root.select('.bars')
  }

  render(rawData) {
    if (rawData) {
      this.rawData = rawData
      this.data = rawData
    }
    this.updateReviewText()
    this.updateBricks()
    this.updateBars()
  }

  initBricks() {

    let labelWidth = 260
    let labels = this.data['label_names']
    let level1 = {}
    labels.forEach(function(item) {
      let tmp = item.split('_')
      let name = tmp.shift()
      let name2 = tmp.join(' ')
      if (name2 == 'distance from business district') {
        name2 = 'distance from biz district'
      } else if (name2 == 'willing to consume again') {
        name2 = 'willing to return'
      }
      level1[name] = level1[name] || []
      level1[name].push(name2)
    })
    let level1_keys = Object.keys(level1)
    let level1_y = [0]
    level1_keys.forEach((d, i) => {
      let height = level1[d].length * 30
      level1_y[i+1] = level1_y[i] + height
    })
  
    // Add text labels
    let level1_g = this.bricks.selectAll()
      .data(level1_keys).enter()
        .append('g')
        .attr('class', 'level1')
        .attr('transform', (d, i) => `translate(0, ${level1_y[i] + 20})`)

    level1_g.append('line')
      .attr('x1', 0)
      .attr('x2', labelWidth + 10)
      .attr('y1', 0)
      .attr('y2', 0)
      .style('stroke', '#eee')
      .style('stroke-width', 1)
    this.bricks.append('line')
      .attr('x1', 0)
      .attr('x2', labelWidth + 10)
      .attr('y1', 620)
      .attr('y2', 620)
      .style('stroke', '#eee')
      .style('stroke-width', 1)

    level1_g.append('text')
      .attr('x', 85)
      .attr('y', 8)
      .attr('class', 'level1-label')
      .attr('text-anchor', 'end')
      .attr('alignment-baseline', 'hanging')
      .text((d) => d)
    
    level1_g.each(function(d) {
      d3.select(this).selectAll()
        .data(level1[d]).enter()
          .append('g')
          .attr('class', 'level2')
          .attr('transform', (d, i) => `translate(0, ${i * 30})`)
          .append('text')
          .attr('x', labelWidth)
          .attr('y', 8)
          .attr('text-anchor', 'end')
          .attr('alignment-baseline', 'hanging')
          .text((d) => d)
    })

  }

  buildBricks(probas, variant) {

    let xoffset = variant == 'predicted' ? 270 : 542;
    let fullwidth = 270;
    let n_dim = probas[0].length
    let cmap = ['Gainsboro', 'IndianRed', 'LightBlue', 'MediumSeaGreen']
    let xScale = d3.scaleLinear().domain([0, 1]).range([0, fullwidth])
    let layers = d3.stack().keys(d3.range(n_dim)).order(d3.stackOrderNone)
      .offset(d3.stackOffsetNone)(probas)

    let container = this.bricks.selectAll(`g.${variant}-bars`)

    if (!container.size()) {
      container = this.bricks.append('g')
        .attr('class', `${variant}-bars`)
        .attr('transform', `translate(${xoffset},20)`)

      // Add title text
      container.append('text')
        .text(variant)
        .attr('x', fullwidth / 2)
        .attr('y',  -10)
        .attr('text-anchor', 'middle')

      // add the stacked bar layers
      container.selectAll('g.layer')
        .data(layers).enter().append('g')
          .attr('class', 'layer')
          .style('fill', (d, i) => cmap[i])
        .selectAll('rect')
        .data((d) => d).enter()
          .append('rect')
          .attr('x', (d) => xScale(d[0]))
          .attr('y', (d, i) => i * 30)
          .attr('width', (d) => xScale(d[1] - d[0]))
          .attr('height', 29)
    } else {
      container.selectAll('g.layer').each(function(d, i) {
        d3.select(this).selectAll(`rect`)
        .data(layers[i])
          .transition()
          .duration(400)
          .attr('x', (d) => xScale(d[0]))
          .attr('y', (d, i) => i * 30)
          .attr('width', (d) => xScale(d[1] - d[0]))
          .attr('height', 29)
      })
    }

  }

  updateBricks() {
    let level1_g = this.bricks.selectAll('g.level1')
    if (level1_g.size() == 0) {
      this.initBricks()
    }

    // Add bars.
    // Proba is a 3D Array:
    //  [
    //     [ [prob_-2, prob_-1, ...], record2, ...],
    //     aspect2,
    //     ...
    //  ]
    let probas, true_probas;
    if (this.data['probas']) {
      probas = this.data['probas'].map((rows) => rows[0]);
    } else {
      probas = d3.range(20).map((i) => [1, 0, 0, 0]);
    }
    let label2idx = {'-2': 0, '-1': 1, '0': 2, '1': 3};
    this.buildBricks(probas, 'predicted')

    if (this.data['true_labels']) {
      true_probas = this.data['true_labels'][0].map((val) => {
        let x = [0, 0, 0, 0];
        x[label2idx[val]] = 1;
        return x
      })
    } else {
      true_probas = d3.range(20).map((i) => [1, 0, 0, 0]);
    }
    this.buildBricks(true_probas, 'actual')
  }

  updateBars() {

  }

  updateReviewText() {
    let data = this.data
    this.html('.review-id', data.review.id);
    this.html('.review-text', data.review.content_html);
    this.html('.filter-results', data.filter_results);
    let elem = this.$('.review-text');
    if (this.hasClass('is-folded')) {
      // review text height only increases, this is for avoiding
      // jumpy behavior.
      let origHeight = Math.min(elem.clientHeight, 200); // but don't grow too tall
      elem.style.height = '';
      let newHeight = Math.max(elem.clientHeight, origHeight)
      elem.style.height = origHeight + 'px';
      setTimeout(function() {
        elem.style.height = newHeight + 'px';
      }, 60)
    }
  }

}

export default SingleReviewChart;
