
import { $, debounce } from "./helpers.js"

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

    // cancel existing request
    if (this.t_loading != null) {
      clearTimeout(this.t_loading);
    }
    if (this.req && this.req.abort) {
      this.req.abort();
    }

    this.enterLoading()
    this.req = this.fetchData(qs).then((res) => {
      this.render(res);
      this.exitLoading()
    })
  }

  enterLoading() {
    this.t_loading = setTimeout(() => {
      this.addClass('is-loading')
    }, 200);
  }

  exitLoading() {
    if (this.t_loading != null) {
      clearTimeout(this.t_loading);
    }
    this.removeClass('is-loading')
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
      .on('resize', debounce(this.render.bind(this), 100))
      .on('popstate', () => {
        this.fetchAndUpdate(location.search, false)
      })

    this.root = d3.select(elem)
    this.root.selectAll('form').on('submit', () => {
      this.fetchAndUpdate()
      d3.event.preventDefault()
    })
    let self = this
    this.root.selectAll('.foldable').on('click', function() {
      self.toggleClass('folded', this)
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
