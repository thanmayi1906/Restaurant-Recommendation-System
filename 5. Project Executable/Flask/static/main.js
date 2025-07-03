document.addEventListener("DOMContentLoaded", function () {
    const form = document.querySelector("form");
    const restaurantInput = document.querySelector("input[name='restaurant']");
    const scenarioSelect = document.querySelector("select[name='scenario']");

    form.addEventListener("submit", function (event) {
        // Validate restaurant name
        if (!restaurantInput.value.trim()) {
            alert("Please enter a restaurant name.");
            event.preventDefault();
            return;
        }

        // Validate scenario selection
        if (!scenarioSelect.value) {
            alert("Please select a scenario.");
            event.preventDefault();
            return;
        }

        // Optional: Show loading indicator
        const submitBtn = form.querySelector("button[type='submit']");
        submitBtn.disabled = true;
        submitBtn.textContent = "Loading...";
    });
});