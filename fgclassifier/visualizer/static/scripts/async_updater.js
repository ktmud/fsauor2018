
import { $ } from "./helpers.js"

/**
 * Base class to update the DOM asynchroneous
 */
class AsyncUpdater {

  constructor(elem) {
    // Please udpate these in subclasses
    this.fields = null;  // input/select fields this udpater depends on
    this.endpoint = null;  // API endpoint
    this.rawData = null;   // cached raw data from last request
    this.data = null;  // processed data ready for updating content
    this.prepare(elem)
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
    this.root = d3.select(elem)
    this.root.select('form').on('submit', () => {
      this.fetchAndUpdate()
      d3.event.preventDefault()
    })
  }

  render(rawData) {}

  $(selector) {
    return this.root.select(selector).node()
  }

  html(selector, content) {
    this.root.selectAll(selector).each(function() {
      this.innerHTML = content;
    })
  }

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

export default AsyncUpdater;
