function layout_obj(prob_name, n_exec)
    tickvals_x = formatting_tickvals_10power(1:1000)
    tickvals_y = formatting_tickvals_10power(1e-8:1e9)
    layout_obj = Layout(title = L"\huge{(f + h)(x_j) \quad vs. \quad epochs}",
                        xaxis = attr(
                            #title_text = "epoch",
                            title_standoff=50,
                            tickmode = "array",
                            tickangle = 0,
                            tickvals = tickvals_x,
                            ticktext = formatting_ticktext_10power(tickvals_x)
                        ),

                        yaxis =attr(
                                #title_text = "Exact f+h",
                                showexponent = "all",
                                exponentformat = "power",
                                title_standoff=50,
                                tickvals = tickvals_y,
                                ticktext = formatting_ticktext_10power(tickvals_y)
                            ),

                        xaxis_type="log",
                        yaxis_type="log",
                        template="simple_white",
                        legend = attr(
                            xanchor="right",
                            bgcolor="rgba(255,255,255,.6)",
                            font=attr(size = 35)
                        ),
                        font=attr(size = 25))
    return layout_obj
end

function layout_metr(prob_name, n_exec)
    tickvals_x = formatting_tickvals_10power(1:1000)
    tickvals_y = formatting_tickvals_10power(1e-8:1e9)
    layout_metr = Layout(title = L"\huge{\sqrt{\xi^*_{cp}(x_j,\nu_j^{-1})/\nu_j} \quad vs. \quad epochs}",
                        xaxis = attr(
                            #title_text = "epoch",
                            title_standoff=50,
                            tickmode = "array",
                            tickangle = 0,
                            tickvals = tickvals_x,
                            ticktext = formatting_ticktext_10power(tickvals_x)
                        ),

                        yaxis =attr(
                            #title_text = "√ξcp/ν",
                            showexponent = "all",
                            exponentformat = "power",
                            title_standoff=50,
                            tickvals = tickvals_y,
                            ticktext = formatting_ticktext_10power(tickvals_y)
                        ),

                        xaxis_type="log",
                        yaxis_type="log",
                        template="simple_white",
                        legend = attr(
                            x = 0,
                            y = 0,
                            bgcolor="rgba(255,255,255,.6)",
                            font=attr(size = 35)
                        ),
                        font=attr(size = 18))
    return layout_metr
end

function layout_mse(prob_name, n_exec)
    tickvals_x = formatting_tickvals_10power(1:1000)
    tickvals_y = formatting_tickvals_10power(1e-8:1e9)
    layout_mse = Layout(title = L"\huge{MSE \quad vs. \quad epochs}",
                        xaxis = attr(
                            #title_text = "epoch",
                            title_standoff=50,
                            tickmode = "array",
                            tickangle = 0,
                            tickvals = tickvals_x,
                            ticktext = formatting_ticktext_10power(tickvals_x)
                        ),
                        yaxis =attr(
                            #title_text = "MSE",
                            showexponent = "all",
                            exponentformat = "power",
                            title_standoff=50,
                            tickvals = tickvals_y,
                            ticktext = formatting_ticktext_10power(tickvals_y)
                        ),

                        xaxis_type="log",
                        yaxis_type="log",
                        template="simple_white",
                        legend = attr(
                            xanchor="right",
                            bgcolor="rgba(255,255,255,.6)",
                            font=attr(size = 35)
                        ),
                        font=attr(size = 25))
    return layout_mse
end