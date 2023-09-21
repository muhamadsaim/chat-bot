document.addEventListener("DOMContentLoaded", function () {
    const robot = document.querySelector(".robot");
    const sayHiButton = document.getElementById("sayHi");

    sayHiButton.addEventListener("click", function () {
        alert("Hi!");
    });
});
