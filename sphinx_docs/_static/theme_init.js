/* Force Light Mode as default if no preference is saved */
(function () {
    const theme = localStorage.getItem("theme");
    if (!theme) {
        // Do not force it if system is already set, or do force it?
        // User said "keep original light mode by default". 
        // This usually means ignore system setting initially.
        // localStorage.setItem("theme", "light");
    }
})();

/* Add Edit Button next to Theme Toggle */
function addEditButton() {
    console.log("LeRobot: addEditButton function called");
    
    // Determine filename
    let path = window.location.pathname;
    let filename = path.split("/").pop();
    
    // Handle root path or index
    if (!filename || filename === "" || filename === "index.html") {
        filename = "index.md";
    } else {
        filename = filename.replace(".html", ".md");
    }
    
    console.log("LeRobot: Current page filename:", filename);
    
    // Create the edit button element
    function createEditButton() {
        const editBtn = document.createElement("a");
        editBtn.className = "lerobot-edit-btn";
        editBtn.title = "Edit: " + filename;
        editBtn.style.marginLeft = "8px";
        editBtn.style.display = "inline-flex";
        editBtn.style.alignItems = "center";
        editBtn.style.justifyContent = "center";
        editBtn.style.width = "36px";
        editBtn.style.height = "36px";
        editBtn.style.padding = "0";
        editBtn.style.cursor = "pointer";
        editBtn.style.textDecoration = "none";
        editBtn.style.color = "inherit";
        editBtn.style.border = "none";
        editBtn.style.background = "transparent";
        
        // Pencil Icon SVG
        editBtn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path></svg>`;
        
        // Editor URL (Port 5000)
        let host = window.location.hostname;
        if (!host || host === "") {
            host = "localhost";
        }
        editBtn.href = `http://${host}:5000/edit/${filename}`;
        editBtn.target = "_blank";
        
        return editBtn;
    }
    
    // Strategy 1: Add after theme toggle buttons
    const themeButtons = document.querySelectorAll(".theme-toggle");
    console.log("LeRobot: Found " + themeButtons.length + " theme toggle buttons");
    
    themeButtons.forEach((btn, index) => {
        // Skip if we already added the edit button
        if (btn.nextElementSibling && btn.nextElementSibling.classList.contains("lerobot-edit-btn")) {
            return;
        }
        
        const editBtn = createEditButton();
        btn.parentNode.insertBefore(editBtn, btn.nextSibling);
        console.log("LeRobot: Added edit button after theme button " + index);
    });
    
    // Strategy 2: If no theme buttons found, add to header-right or any header-like element
    if (themeButtons.length === 0) {
        console.log("LeRobot: No theme buttons found, looking for header containers...");
        
        // Try to find header-right div
        let headerRight = document.querySelector(".header-right");
        if (!headerRight) {
            headerRight = document.querySelector("header");
        }
        if (!headerRight) {
            headerRight = document.querySelector(".mobile-header");
        }
        
        if (headerRight) {
            // Check if edit button already exists
            if (!headerRight.querySelector(".lerobot-edit-btn")) {
                const editBtn = createEditButton();
                headerRight.appendChild(editBtn);
                console.log("LeRobot: Added edit button to header element");
            }
        } else {
            console.log("LeRobot: Could not find suitable header element");
        }
    }
}

// Wait for DOM to be ready
if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", addEditButton);
} else {
    // DOM is already loaded
    setTimeout(addEditButton, 100);
}

// Also try again after a slight delay in case elements load dynamically
setTimeout(addEditButton, 500);
setTimeout(addEditButton, 1000);
