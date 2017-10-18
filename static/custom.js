function toggle_left_panel() {
    var left_panel = document.getElementById("left_panel");
    var right_panel = document.getElementById("right_panel");

    if (left_panel.style.display === "none") {
        left_panel.style.display = "block";

        right_panel.classList.add('col-md-6');
        right_panel.classList.remove('col-md-12');


    } else {
        left_panel.style.display = "none";

        right_panel.classList.add('col-md-12');
        right_panel.classList.remove('col-md-6');


    }
}
