function layout3d(name, camera_settings)
    layout = Layout(scene = attr(
                xaxis = attr(
                     backgroundcolor="rgb(255, 255, 255)",
                     title_text = "",
                     gridcolor="white",
                     showbackground=false,
                     zerolinecolor="white",
                     tickfont=attr(size=0, color="white")),

                yaxis = attr(
                    backgroundcolor="rgb(255, 255, 255)",
                    title_text = "",
                    gridcolor="white",
                    showbackground=false,
                    zerolinecolor="white",
                    tickfont=attr(size=0, color="white")),

                zaxis = attr(
                    backgroundcolor="rgb(255, 255, 255)",
                    title_text = "",
                    gridcolor="white",
                    showbackground=false,
                    zerolinecolor="white",
                    tickfont=attr(size=0, color="white")),
                    margin=attr(
                        r=10, l=10,
                        b=10, t=10),
                    aspectmode = "manual",
                    showlegend = false
                    ),
                    scene_camera = camera_settings[name]
        )
    return layout
end