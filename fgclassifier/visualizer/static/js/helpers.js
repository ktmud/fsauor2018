export const initHoverTips = (bars, renderTooltip) => {
  let tooltip = d3.select('body')
    .append('div').attr('class', 'tooltip')

  let lastBar = -1;
  let t_hideTooltip = 0;

  bars.on("mouseover", function(d, i) {

    clearTimeout(t_hideTooltip);

    let closestGroup = d3.select(this).node().closest('g')
    let groupId = d3.select(closestGroup).attr('group-id')
    let myId = groupId + String(i)
    if (myId !== lastBar) {
      lastBar = myId 
      let text = renderTooltip(d, groupId, i)
      if (text) {
        tooltip.html(text)
        tooltip.attr('class', 'tooltip active')
          .style('transition', 'none')
      }
    }
  }).on("mouseout", () => {
    t_hideTooltip = setTimeout(() => {
      tooltip.attr('class', 'tooltip')
        .style('transition', '.2s all')
    })
  }).on("mousemove", function mousemove(d, i) {
    let [xm, ym] = [d3.event.pageX + 12, d3.event.pageY + 6];
    let tooltipWidth = tooltip.node().clientWidth

    if (xm + tooltipWidth + 20 > window.innerWidth) {
      xm = xm - tooltipWidth - 20
    }
    tooltip.style("transform", "translate(" + xm + "px," + ym + "px)");
  });
}

export const debounce = (fn, time) => {
  let timeout;
  return function () {
    const functionCall = () => fn.apply(this, arguments);
    clearTimeout(timeout);
    timeout = setTimeout(functionCall, time);
  }
}

export const $ = (id) => document.querySelector(id);
