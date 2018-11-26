// data
let data = null;
let $ = (id) => document.querySelector(id)

class AsyncUpdater {

  constructor(elem) {
    // Please udpate these in subclasses
    this.fields = null;  // input/select fields this udpater depends on
    this.endpoint = null;  // API endpoint
    this.rawData = null;   // cached raw data from last request
    this.prepare(elem);
  }

  fetchAndUpdate(qs) {
    this.fetchData(qs).then(this.render.bind(this));
  }

  /**
   * Prepare canvas and init events
   */
  prepare(elem) {
    d3.select(window)
      .on('resize', this.render.bind(this))
      .on('popstate', () => {
        this.fetchAndUpdate(location.search, false)
      })
    d3.select('form').on('submit', () => {
      this.fetchAndUpdate()
      d3.event.preventDefault()
    })
    d3.select('.prev-seed').on('click', () => {
      let elem = $('#inp-seed')
      elem.value = Math.max(parseInt(elem.value, 0) - 1, 0)
      this.fetchAndUpdate()
    })
    d3.select('.next-seed').on('click', () => {
      let elem = $('#inp-seed')
      elem.value = Math.min(parseInt(elem.value, 0) + 1, 99999999)
      this.fetchAndUpdate()
    })
  }

  render(rawData) {}

  fetchData(qs, updateHistory) {
    let p = (this.params = {});
    if (qs) {
      let params = new URLSearchParams(qs)
      for (let [k, val] of params.entries()) {
        let elem = $('#sel-' + k) || $('#inp-' + k)
        if (elem) {
          elem.value = val
          p[k] = val
        }
      }
    }

    let optNames = this.fields
    optNames.forEach((k) => {
      let elem = $('#sel-' + k) || $('#inp-' + k)
      if (elem) {
        p[k] = elem.value;
      }
    });
    let new_qs = [];
    for (let k of Object.keys(p)) {
      new_qs.push(`${k}=${p[k]}`)
    }
    new_qs = '?' + new_qs.join('&')

    if (updateHistory !== false && new_qs != qs) {
      window.history.pushState(null, null, new_qs);
    }

    // update form class, to hide certain hyperparameters
    d3.select('form').attr('class', `form model-${p.model}`)
    return d3.json(`${this.endpoint}${new_qs}`);
  }
}

class ReviewSelector extends AsyncUpdater {

  constructor(elem) {
    this.fields = ['dataset', 'keyword', 'seed', 'fm', 'clf']
    this.endpoint = '/pick_review'
    this.prepare(elem);
  }

}

class ModelSelector extends AsyncUpdater {

  constructor(elem) {
    this.fields = ['dataset', 'keyword', 'seed', 'fm', 'clf']
    this.endpoint = '/pick_review'
    this.prepare(elem);
  }

}
// init and render
let reviewSelector = new ReviewSelector('.pick-review');
let modelSelector = new ModelSelector('.pick-model');
chart.fetchAndUpdate(location.search, false);
