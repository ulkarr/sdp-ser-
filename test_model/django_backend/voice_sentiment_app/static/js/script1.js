document.addEventListener("DOMContentLoaded", function() {
    document.querySelectorAll(".images-hoc img").forEach(function(img) {

        img.addEventListener("mousemove", function() {
            scaleImage(img, 1.3);
        });

        img.addEventListener("mouseleave", function() {
            scaleImage(img, 1);
        });
    });

    function scaleImage(img, scale) {
        img.style.transform = "scale(" + scale + ")";
        img.style.transition = "transform 0.5s";
    }
});

// sources that helped: https://developer.mozilla.org/en-US/docs/Web/API/Document_Object_Model/Introduction
// https://www.youtube.com/watch?v=to2xUC1FTNM&ab_channel=InventionTricks
// https://developer.mozilla.org/en-US/docs/Web/API/Document/querySelectorAll