/**
 * Single Review predictions
 */
import {
  $,
  initHoverTips
} from "./helpers.js"
import AsyncUpdater from './async_updater.js'

const labelWidth = 230;
const barHeight = 29;
const label2idx = {
  '-2': 0,
  '-1': 1,
  '0': 2,
  '1': 3
};
const labeltext = ['Not Mentioned', 'Negative', 'Neutral', 'Positive'];

const label2probas = (labels) => {
  return labels.map((row) => {
    return row.map((val) => {
      let x = [0, 0, 0, 0];
      x[label2idx[val]] = 1;
      return x
    })
  })
}
const labelcounts = (counts) => {
  return counts.map((row) => {
    let x = [0, 0, 0, 0];
    for (let key of Object.keys(row)) {
      x[label2idx[key]] = row[key];
    }
    return x
  })
}

class SingleReviewChart extends AsyncUpdater {

  constructor(elem) {
    super(elem)
    // to make a prediction, we need all parameters
    this.fields = ['dataset', 'keyword', 'seed', 'fm', 'clf']
    this.endpoint = '/predict'

    this.root.selectAll('.prev-seed').on('click', () => {
      let elem = $('#inp-seed')
      elem.value = Math.max(parseInt(elem.value, 0) - 1, 0)
      this.fetchAndUpdate()
    })
    this.root.selectAll('.next-seed').on('click', () => {
      let elem = $('#inp-seed')
      elem.value = Math.min(parseInt(elem.value, 0) + 1, 99999999)
      this.fetchAndUpdate()
    })
    this.root.selectAll('select, input').on('change', () => {
      this.fetchAndUpdate()
    })

    let reviewText = this.root.selectAll('.review-text').on('mouseup', () => {
      // current sentence node
      let curNode = d3.event.target
      if (!curNode.classList.contains('sentence')) {
        curNode = curNode.closest('.sentence')
      }
      // all sentence nodes
      let nodes = reviewText.selectAll('.sentence').nodes()

      // activate selected nodes
      let sel = window.getSelection()
      let text = '' // selected text

      if (curNode && curNode.classList.contains('active') && sel.toString() === '') {
        this.fetchAndUpdate()
        return;
      }

      nodes.forEach((node) => {
        // deactivate all nodes
        node.classList.remove('active')
        node.classList.remove('is-first')
        node.classList.remove('is-last')
      })

      let filtered = nodes.filter((node) => {
        return node == curNode || sel.containsNode(node, true)
      })
      let last = filtered.length - 1

      filtered.forEach((node, i) => {
        node.classList.add('active')
        if (i == 0) {
          node.classList.add('is-first')
        }
        if (i == last) {
          node.classList.add('is-last')
        }
        text += node.innerText;
      })

      sel.removeAllRanges()

      this.fetchAndUpdate({
        params: {
          text
        },
        endpoint: '/predict_text',
        fields: ['fm', 'clf']
      })
    })
  }

  prepare(elem) {
    super.prepare(elem)
    this.bricks = this.root.select('.bricks')
    this.bars = this.root.select('.bars')
  }

  render(rawData) {
    if (rawData) {
      this.data = {
        ...this.data,
        ...rawData
      }
      this.data.review = rawData.review;
      this.data.n_correct_labels_html = rawData.n_correct_labels_html;
      this.data.has_true_labels = !!rawData.true_labels;

      // only update text when there's new data passed in
      if (this.data.review) {
        this.updateReviewText()
      }
      this.updateCorrectCount()
      this.updateOverallBars()
    }
    if (this.data) {
      this.updateBricks()
    }
  }

  initBricks() {

    let labels = this.data['label_names']
    let level1 = {}
    labels.forEach(function (item) {
      let tmp = item.split('_')
      let name = tmp.shift()
      let name2 = tmp.join(' ')
      if (name2 == 'distance from business district') {
        name2 = 'dist from biz district'
      } else if (name2 == 'willing to consume again') {
        name2 = 'willing to return'
      } else if (name2 == 'cleaness') {
        name2 = 'cleanness'
      }
      level1[name] = level1[name] || []
      level1[name].push(name2)
    })
    let level1_keys = Object.keys(level1)
    let level1_y = [0]
    level1_keys.forEach((d, i) => {
      let height = level1[d].length * (barHeight + 1)
      level1_y[i + 1] = level1_y[i] + height
    })

    // Add text labels
    let level1_g = this.bricks.selectAll()
      .data(level1_keys).enter()
      .append('g')
      .attr('class', 'level1')
      .attr('transform', (d, i) => `translate(0, ${level1_y[i] + 20})`)

    level1_g.append('line')
      .attr('x1', 0).attr('x2', labelWidth + 10)
      .attr('y1', 0).attr('y2', 0)
      .style('stroke', '#eee').style('stroke-width', 1)
    this.bricks.append('line')
      .attr('x1', 0).attr('x2', labelWidth + 10)
      .attr('y1', 620).attr('y2', 620)
      .style('stroke', '#eee').style('stroke-width', 1)

    level1_g.append('text')
      .attr('x', 85).attr('y', 8)
      .attr('class', 'level1-label')
      .attr('text-anchor', 'end')
      .attr('alignment-baseline', 'hanging')
      .text((d) => d)

    level1_g.each(function (d) {
      d3.select(this).selectAll()
        .data(level1[d]).enter()
        .append('g').attr('class', 'level2')
        .attr('transform', (d, i) => `translate(0, ${i * (barHeight + 1)})`)
        .append('text').attr('x', labelWidth).attr('y', 8)
        .attr('text-anchor', 'end')
        .attr('alignment-baseline', 'hanging')
        .text((d) => d)
    })

  }

  buildBricks(probas, variant, {
    root,
    fullwidth = 120,
    barheight = barHeight,
    barmargin = 1,
    showtitle = true,
    xoffset = 0,
    yoffset = 0,
    titlex = 0,
    titley = 0,
    alpha = 1,
    titleAnchor = 'middle',
    orient = 'horizontal',
    tooltipTmpl = (d, i) => {
      let dd = d[1] - d[0]
      if (dd < 1) {
        dd = dd.toFixed(2);
      }
      return labeltext[i] + ': ' + dd;
    }
  } = {}) {
    root = root || this.bricks;

    let n_dim = probas[0].length
    // let cmap = ['Gainsboro', 'IndianRed', 'LightBlue', 'MediumSeaGreen'];
    let cmap = ['Gainsboro', '#d7191c', '#abd9e9', '#1a9641'];
    let domainMax = probas[0].reduce((a, b) => a + b, 0);
    let variant_cls = `${variant.replace(' ', '-')}-bars`
    let container = root.selectAll(`g.${variant_cls}`)
    let isInit = !container.size()

    if (isInit) {
      container = root.append('g')
        .attr('class', `brick-bars ${variant_cls}`)
        .attr('transform', `translate(${xoffset},${yoffset})`)
      if (showtitle) {
        // Add title text
        container.append('text')
          .text(variant)
          .attr('x', titlex)
          .attr('y', titley)
          .attr('fill-opacity', alpha)
          .attr('text-anchor', titleAnchor)
      }
    }

    if (orient == 'vertical') {
      let yScale = d3.scaleLinear().domain([0, domainMax]).range([0, barHeight])
      let xScale = d3.scaleLinear().domain([0, 3]).range([0, fullwidth * (n_dim - 1) / n_dim])
      if (isInit) {
        let bars = container.selectAll('g.layer')
          .data(probas).enter().append('g')
          .attr('class', 'layer')
          .attr('group-id', (d, i) => i)
          .attr('transform', (d, i) => `translate(0,${i * (barheight + barmargin)})`)
          .selectAll('rect')
          .data((d) => d).enter()
          .append('rect')
          .style('fill', (d, i) => cmap[i])
          .attr('x', (d, i) => xScale(i))
          .attr('height', (d) => yScale(d))
          .attr('y', (d) => barheight - yScale(d))
          .attr('width', fullwidth / n_dim)

        initHoverTips(bars, function (d, i, j) {
          return labeltext[j] + ': ' + d
        })
      } else {

      }
      return
    }

    // Show as Horizontal stacked bars -------

    let xScale = d3.scaleLinear().domain([0, domainMax]).range([0, fullwidth]);
    let layers = d3.stack().keys(d3.range(n_dim))
      .order(d3.stackOrderNone)
      .offset(d3.stackOffsetNone)(probas)

    if (isInit) {
      // add the stacked bar layers
      let bars = container.selectAll('g.layer')
        .data(layers).enter().append('g')
        .attr('class', 'layer')
        .attr('group-id', (d, i) => i)
        .style('fill', (d, i) => cmap[i])
        .selectAll('rect')
        .data((d) => d).enter()
        .append('rect')
        .attr('fill-opacity', alpha)
        .attr('x', (d) => xScale(d[0]))
        .attr('y', (d, i) => i * (barheight + barmargin))
        .attr('width', (d) => xScale(d[1] - d[0]))
        .attr('height', barheight)
      initHoverTips(bars, tooltipTmpl)
    } else {
      container
        .transition().duration(400)
        .attr('transform', `translate(${xoffset},${yoffset})`)
        .selectAll('g.layer').each(function (d, i) {
          d3.select(this).selectAll(`rect`)
            .data(layers[i])
            .transition()
            .duration(400)
            .attr('fill-opacity', alpha)
            .attr('x', (d) => xScale(d[0]))
            .attr('width', (d) => xScale(d[1] - d[0]))
        })
    }

  }

  updateBricks() {
    let level1_g = this.bricks.selectAll('g.level1')
    if (level1_g.size() == 0) {
      this.initBricks()
    }

    let fullwidth = (this.$('.chart').clientWidth - labelWidth - 30) / 2;
    fullwidth = Math.max(80, fullwidth)
    let xoffset = labelWidth + 8
    let yoffset = 20
    let titlex = fullwidth / 2
    let titley = -8
    let configs = {
      fullwidth,
      xoffset,
      yoffset,
      titlex,
      titley
    }

    // Add bars.
    // Proba is a 3D Array:
    //  [
    //     [ [prob_-2, prob_-1, ...], record2, ...],
    //     aspect2,
    //     ...
    //  ]
    let probas, predict_label_probas, true_probas;
    if (this.data['predict_labels']) {
      predict_label_probas = label2probas(this.data['predict_labels'])[0];
    }
    if (this.data['probas']) {
      probas = this.data['probas'].map((rows) => rows[0]);
    } else if (predict_label_probas) {
      probas = predict_label_probas
    } else {
      probas = d3.range(20).map((i) => [1, 0, 0, 0]);
    }

    this.buildBricks(probas, 'predicted', configs)
    this.buildBricks(predict_label_probas, 'P', {
      ...configs,
      titlex: barHeight / 2,
      xoffset: xoffset + fullwidth + 5,
      fullwidth: barHeight,
      tooltipTmpl: (d, i) => {
        return `Predicted:<br>${labeltext[i]}`
      }
    })

    if (this.data['true_labels']) {
      true_probas = label2probas(this.data['true_labels'])[0];
    } else {
      true_probas = d3.range(20).map((i) => [1, 0, 0, 0]);
    }
    this.buildBricks(true_probas, 'A', {
      ...configs,
      titlex: 12,
      xoffset: xoffset + fullwidth + barHeight + 6,
      fullwidth: barHeight,
      alpha: this.data.has_true_labels ? 1 : 0.4,
      tooltipTmpl: (d, i) => {
        return `Actual:<br>${labeltext[i]}`
      }
    })

    if (this.data['global_count']) {
      let margin = 12
      let width = fullwidth - barHeight * 2 - margin
      this.buildBricks(this.data['global_count'], 'global distribution', {
        ...configs,
        xoffset: xoffset + fullwidth + 2 * barHeight + margin,
        titlex: width / 2,
        fullwidth: width,
        alpha: 0.4,
      })
    }
  }

  updateCorrectCount() {
    let html = this.data.n_correct_labels_html
    html = html || 'We don\'t know the true labels.'
    this.html('.correct-count', html);
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
      setTimeout(function () {
        elem.style.height = newHeight + 'px';
      }, 60)
    }
  }

  updateOverallBars(show_actual) {
    let root = d3.select(this.$('.overall-bars'))
    let counts;
    let configs = {
      root,
      barheight: 16,
      fullwidth: 100,
      xoffset: 70,
      yoffset: 0,
      titlex: -8,
      titley: 12,
      titleAnchor: 'end'
    }
    counts = labelcounts(this.data.predict_label_counts);
    this.buildBricks(counts, 'predicted', configs)
    if (this.data.true_label_counts) {
      counts = labelcounts(this.data.true_label_counts);
    }
    configs = {
      ...configs,
      yoffset: 17,
    }
    this.buildBricks(counts, 'actual', configs)
  }

}

export default SingleReviewChart;
