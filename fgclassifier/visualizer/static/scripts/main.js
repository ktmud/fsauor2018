// data
let data = null;

class RocChart {
  constructor(elem) {
    this.prepare(elem);
    d3.selectAll("select,#sel-c").on("change", this.fetchAndUpdate.bind(this));
    d3.select(window)
      .on('resize', this.render.bind(this))
      .on('popstate', () => {
        this.fetchAndUpdate(location.search, false)
      })
    d3.select('form').on('submit', () => {
      d3.event.preventDefault()
    })
  }

  fetchAndUpdate(qs) {
    this.fetchData(qs).then(this.render.bind(this));
  }

  /**
   * Prepare canvas
   */
  prepare(elem) {
    const chart = (this.chart = d3.select(elem));
    const svg = (this.svg = chart.append("svg"));

    let x = (this.x = d3
      .scaleLinear()
      .domain([0, 1])
      .nice());

    let y = (this.y = d3
      .scaleLinear()
      .domain([0, 1])
      .nice());

    this.line = d3
      .line()
      .defined((d) => !isNaN(d.fpr))
      .x((d) => x(d.fpr))
      .y((d) => y(d.tpr));

    this.xAxis = d3.axisBottom(x).tickSizeOuter(0).ticks(10);
    this.yAxis = d3.axisLeft(y).tickSizeOuter(0).ticks(10);

    this.xlabel = svg
      .append("text")
      .text("False Postive Rate")
      .style("text-anchor", "middle");
    this.ylabel = svg
      .append("text")
      .text("True Postive Rate")
      .style("text-anchor", "middle");

    this.title = svg
      .append("text")
      .attr("class", "title")
      .text("ROC")
      .style("text-anchor", "middle")

    svg
      .append("path")
      .attr("class", "line data-line")
      .attr("fill", "none")
      .attr("stroke", "steelblue")

    // diagonal reference line
    svg
      .append("path")
      .attr("class", "line ref-line")
      .attr("fill", "none")
      .attr("stroke", "orange")

    svg.append("g").attr("class", "x axis");
    svg.append("g").attr("class", "y axis");

    return svg.node();
  }

  render(rawData) {
    if (rawData) {
      data = rawData["fpr"].map((d, i) => {
        return { fpr: d, tpr: rawData["tpr"][i] };
      });
      data.auc = rawData.auc;
      data.params = rawData.params;
    }

    let p = data.params;
    let title = `ROC - ${p.preprocessing}, ${p.model}`;
    if (p.C && p.kernel) {
      title += `(C=${p.C}, kernel=${p.kernel})`;
    } else if (p.C) {
      title += `(C=${p.C})`;
    }

    const chart = this.chart;
    let margin = { top: 60, right: 20, bottom: 10, left: 22 };
    const width = Math.min(500, chart.node().offsetWidth);
    const height = width;
    let xAxisHeight = 40;
    let yAxisWidth = 38;

    let svg = chart.select("svg");

    this.x.range([margin.left + yAxisWidth, width - margin.right]);
    this.y.range([height - margin.bottom - xAxisHeight, margin.top]);

    svg
      .select("path.data-line")
      .datum(data)
      .transition()
      .duration(700)
      .attr("d", this.line);
    svg
      .select("path.ref-line")
      .style("stroke-dasharray", "3, 3")  // <== This line here!!
      .datum([{tpr: 0, fpr: 0}, {tpr: 1, fpr: 1}])
      .attr("d", this.line)

    svg.transition().duration(200).attr("width", width).attr("height", height);

    svg
      .select("g.x.axis")
      .call(this.xAxis)
      .attr(
        "transform",
        `translate(0, ${height - margin.bottom - xAxisHeight})`
      );

    svg
      .select("g.y.axis")
      .call(this.yAxis)
      .attr("transform", `translate(${margin.left + yAxisWidth}, 0)`);

    this.xlabel.attr(
      "transform",
      `translate(${margin.left + width / 2} , ${height - margin.bottom})`
    );
    this.ylabel.attr(
      "transform",
      `rotate(-90), translate(${-margin.top - (height - margin.top) / 2} , ${
        margin.left
      })`
    );
    this.title
      .text(title)
      .attr(
        "transform",
        `translate(${yAxisWidth + margin.left + (width - yAxisWidth - margin.left)/2}, 30)`
      );
  }

  fetchData(qs, updateHistory) {
    let p = (this.params = {});
    if (qs) {
      let params = new URLSearchParams(qs)
      for (var [k, val] of params.entries()) {
        let elem = document.getElementById("sel-" + k)
        if (elem) {
          elem.value = val
          p[k] = val
        }
      }
    }
  
    let optNames = ["preprocessing", "model", "c", "kernel"];
    optNames.forEach((k) => {
      p[k] = document.getElementById("sel-" + k).value;
    });

    // force preprocessing for SVM model
    if (p.model == 'svm') {
      d3.select('option[value="none"]').attr('disabled', true)
    } else {
      d3.select('option[value="none"]').attr('disabled', null)
    }
    if (p.model == 'svm' && p.preprocessing == 'none') {
      p.preprocessing = 'standardization'
      d3.select('sel-preprocessing').attr('value', p.preprocessing)
    }

    let new_qs =
      `?preprocessing=${p.preprocessing}&model=${p.model}` +
      `&c=${p.c}&kernel=${p.kernel}`;

    if (updateHistory !== false && new_qs != qs) {
      window.history.pushState(null, null, new_qs);
    }

    // update form class, to hide certain hyperparameters
    d3.select('form').attr('class', `form model-${p.model}` )
    return d3.json(`/roc_curve${new_qs}`);
  }
}

// init and render
let chart = new RocChart("#chart");
chart.fetchAndUpdate(location.search, false);
