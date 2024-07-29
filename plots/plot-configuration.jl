compound = 1
color_scheme = Dict([(1.0, "rgb(255,105,180)"), (.2, "rgb(30,144,255)"), (.05, "rgb(178,34,34)"), (.1, "rgb(255,215,0)"), (.01, 8)])
color_scheme_std = Dict([(1.0, "rgba(255,105,180, .2)"), (.2, "rgba(30,144,255, 0.2)"), (.05, "rgba(178,34,34, 0.2)"), (.1, "rgba(255,215,0, 0.2)"), (.01, 8)])

prob_versions_names = Dict([(1, "mobmean"), (2, "ND-E"), (3, "each-it"), (4, "hybrid"), (5, "ND-S"), (6, "hybrid-acc")])
prob_versions_colors = Dict([(1, "rgb(30,144,255)"), (2, "rgb(255,140,0)"), (3, "rgb(50,205,50)"), (4, "rgb(123,104,238)"), (5, "rgb(50,205,50)"), (6, "rgb(148,0,211)")])
prob_versions_colors_std = Dict([(1, "rgba(30,144,255, 0.2)"), (2, "rgba(255,140,0, 0.2)"), (3, "rgba(50,205,50, 0.2)"), (4, "rgba(123,104,238, 0.2)"), (5, "rgba(250,205,50, .2)"), (6, "rgba(148,0,211, .2)")])

smooth_versions_colors = Dict([(1, "rgb(65,105,225)"), (2, "rgb(255,215,0)"), (3, "rgb(34,139,34)"), (4, "rgb(75,0,130)"), (5, "rgb(34,139,34)"), (6, "rgb(72,61,139)")])
smooth_versions_colors_std = Dict([(1, "rgba(65,105,225, 0.2)"), (2, "rgba(255,215,0, 0.2)"), (3, "rgba(34,139,34, 0.2)"), (4, "rgba(75,0,130, 0.2)"), (5, "rgba(34,139,34, .2)"), (6, "rgba(72,61,139, .2)")])

line_style_sto = Dict([(1.0, "default"), (.2, "dot"), (.1, "dash"), (.05, "dashdot")])
line_style_plm = Dict([(2, "default"), (5, "dash"), (6, "dashdot"), (4, "dot")])

line_style_sto_pgf = Dict([(1.0, "solid"), (.2, "dashdotted"), (.1, "thin"), (.05, "thick"), (.01, "semithick")])
line_style_plm_pgf = Dict([(2, "solid"), (5, "thick"), (6, "thin"), (4, "semithick")])
prob_versions_colors_pgf = Dict([(1, "blue"), (2, "orange"), (3, "yellow"), (4, "pink"), (5, "green"), (6, "purple")])
color_scheme_pgf = Dict([(1.0, "black"), (.2, "rgb(30,144,255)"), (.05, "red"), (.1, "blue"), (.01, "brown")])

Confidence = Dict([("95%", 1.96), ("99%", 2.58)])
conf = "95%"