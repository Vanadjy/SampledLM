function layout(prob_name, n_exec, selected_h)
    layout_obj = Layout(title="$prob_name - $n_exec runs - h = $selected_h-norm",
                        xaxis_title="epoch",
                        xaxis_type="log",
                        yaxis =attr(
                                showexponent = "all",
                                exponentformat = "e"
                            ),
                        yaxis_type="log",
                        yaxis_title="Exact f+h",
                        template="simple_white",
                        legend = attr(
                            xanchor="right",
                            bgcolor="rgba(255,255,255,.4)"
                        ),
                        font=attr(size = 18))
                
    layout_metr = Layout(title="$prob_name - $n_exec runs - h = $selected_h-norm",
                        xaxis_title="epoch",
                        xaxis_type="log",
                        yaxis =attr(
                            showexponent = "all",
                            exponentformat = "e"
                        ),
                        yaxis_type="log",
                        yaxis_title="√ξcp/ν",
                        template="simple_white",
                        legend = attr(
                            x = 0,
                            y = 0,
                            bgcolor="rgba(255,255,255,.4)"
                        ),
                        font=attr(size = 18))

    layout_mse = Layout(title= prob_name == "ijcnn1-ls" ? "PLM - $prob_name - $n_exec runs - h = $selected_h-norm" : "$prob_name - $n_exec runs - h = $selected_h-norm" ,
                        xaxis_title="epoch",
                        xaxis_type="log",
                        yaxis =attr(
                            showexponent = "all",
                            exponentformat = "e"
                        ),
                        yaxis_type="log",
                        yaxis_title="MSE",
                        template="simple_white",
                        legend = attr(
                            xanchor="right",
                            bgcolor="rgba(255,255,255,.4)"
                        ),
                        font=attr(size = 18))
    return layout_obj, layout_metr, layout_mse
end