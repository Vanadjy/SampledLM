function layout(prob_name, n_exec, selected_h)
    layout_obj = Layout(#title= prob_name == "ijcnn1-ls" ? "PLM - $prob_name - $n_exec runs" : "$prob_name - $n_exec runs",
                        xaxis = attr(
                            #title_text = "epoch",
                            title_standoff=50
                        ),

                        yaxis =attr(
                                #title_text = "Exact f+h",
                                showexponent = "all",
                                exponentformat = "e",
                                title_standoff=50
                            ),

                        xaxis_type="log",
                        yaxis_type="log",
                        template="simple_white",
                        legend = attr(
                            xanchor="right",
                            bgcolor="rgba(255,255,255,.6)",
                            font=attr(size = 35)
                        ),
                        font=attr(size = 23))
                
    layout_metr = Layout(title= prob_name == "ijcnn1-ls" ? "PLM - $prob_name - $n_exec runs" : "$prob_name - $n_exec runs",
                        xaxis = attr(
                            #title_text = "epoch",
                            title_standoff=50
                        ),

                        yaxis =attr(
                            #title_text = "√ξcp/ν",
                            showexponent = "all",
                            exponentformat = "e",
                            title_standoff=50
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
                        font=attr(size = 23))

    layout_mse = Layout(#title= prob_name == "ijcnn1-ls" ? "PLM - $prob_name - $n_exec runs" : "$prob_name - $n_exec runs" ,
                        xaxis = attr(
                            #title_text = "epoch",
                            title_standoff=50
                        ),
                        yaxis =attr(
                            #title_text = "MSE",
                            showexponent = "all",
                            exponentformat = "e",
                            title_standoff=50
                        ),

                        xaxis_type="log",
                        yaxis_type="log",
                        template="simple_white",
                        legend = attr(
                            xanchor="right",
                            bgcolor="rgba(255,255,255,.6)",
                            font=attr(size = 35)
                        ),
                        font=attr(size = 23))
    return layout_obj, layout_metr, layout_mse
end