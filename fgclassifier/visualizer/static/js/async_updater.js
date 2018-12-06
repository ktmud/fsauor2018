
import { $, debounce } from "./helpers.js"

/**
 * Base class to update the DOM asynchroneous
 */
class AsyncUpdater {

  constructor(elem, name) {
    // Please udpate these in subclasses
    this.fields = null;  // input/select fields this udpater depends on
    this.endpoint = null;  // API endpoint
    this.rawData = null;   // cached raw data from last request
    this.data = null;  // processed data ready for updating content
    this.name = name || 'updater'
    this.prepare(elem)
    this.loaderDelay = 300;
    this._lastRequest = null;
  }

  fetchAndUpdate(opts) {

    // cancel existing request
    if (this.t_loading != null) {
      clearTimeout(this.t_loading);
    }
    let req = this._lastRequest
    if (req && req.abort) {
      req.abort();
    }

    this.enterLoading()
    req = this._lastRequest = this.fetchData(opts)
    return req.then((res) => {
      // only do the update for the last request
      if (this._lastRequest == req) {
        this.render(res);
        this.exitLoading()
      }
    })
  }

  isParamStale(params) {

  }

  enterLoading() {
    this.t_loading = setTimeout(() => {
      this.addClass('is-loading')
      this.addClass(`is-loading-${this.name}`)
    }, this.loaderDelay);
  }

  exitLoading() {
    if (this.t_loading != null) {
      clearTimeout(this.t_loading);
    }
    this.removeClass('is-loading')
    this.removeClass(`is-loading-${this.name}`)
    this.t_loading = null;
  }

  addClass(cls, root) {
    root = root || this.root.node()
    root.classList.add(cls)
  }

  removeClass(cls, root) {
    root = root || this.root.node()
    root.classList.remove(cls)
  }

  hasClass(cls, root) {
    root = root || this.root.node()
    root.classList.contains(cls)
  }

  toggleClass(cls, root) {
    root = root || this.root.node()
    root.classList.toggle(cls)
  }

  /**
   * Prepare canvas and init events
   */
  prepare(elem) {
    d3.select(window)
      .on(`resize.${this.name}`, debounce(this.render.bind(this), 100))
      .on(`popstate.${this.name}`, () => {
        this.fetchAndUpdate({ qs: location.search }, false)
      })

    this.root = d3.select(elem)
    this.root.selectAll('form').on(`submit.${this.name}`, () => {
      this.fetchAndUpdate()
      d3.event.preventDefault()
    })

    this.root.selectAll('select, input').on(`change.${this.name}`, () => {
      let elem = d3.event.target;
      // if of corresponding fields
      if (~this.fields.indexOf(elem.name)) {
        this.fetchAndUpdate()
      }
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

  fetchData({
    qs,       // query string
    params,   // parameters
    endpoint,  // API endpoint
    fields,   // fields to sent
    updateHistory  // whether to update the history object
  } = {}) {

    let p = params || {};
    this.params = p;

    // Update form control values from query string
    if (typeof qs === 'string') {
      let qsparams = new URLSearchParams(qs)
      for (let [k, val] of qsparams.entries()) {
        let elem = $('#sel-' + k) || $('#inp-' + k)
        if (elem) {
          elem.value = val
          p[k] = val
        }
      }
    }

    // if targeting other endpoint, don't update history
    endpoint = endpoint || this.endpoint
    if (typeof updateHistory == 'undefined' && endpoint != this.endpoint) {
      updateHistory = false;
    }

    // collect values from form controls
    let optNames = fields || this.fields
    optNames.forEach((k) => {
      let elem = $('#sel-' + k) || $('#inp-' + k)
      if (elem) {
        p[k] = elem.value;
      }
    });

    // construct new query strings
    let new_qs = [];
    for (let k of Object.keys(p)) {
      new_qs.push(`${k}=${encodeURIComponent(p[k])}`)
    }
    new_qs = '?' + new_qs.join('&')

    if (updateHistory !== false && new_qs != qs) {
      window.history.pushState(null, null, new_qs);
    }

    return d3.json(`${endpoint}${new_qs}`);
  }
}

export default AsyncUpdater;
