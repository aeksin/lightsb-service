import { refreshAccessToken } from "./refresh_token.js"
import { checkAuth } from "./checkauth.js"
import { logout } from "./logout.js"

document.addEventListener("DOMContentLoaded", async () => {
  // Initially hide auth-related UI elements to prevent any flash until check completes.
  document.querySelectorAll(".unauthorized, .authorized").forEach((el) => (el.style.display = "none"))

  const isAuthenticated = await updateAuthUI()

  const menuItems = document.querySelectorAll(".Menu-item")
  const contentArea = document.querySelector(".Content-area")

  async function updateAuthUI() {
    // Ensure elements are hidden while the auth check is pending.
    document.querySelectorAll(".unauthorized, .authorized").forEach((el) => (el.style.display = "none"))

    const isAuthenticated = await checkAuth()
    document.querySelectorAll(".unauthorized").forEach((el) => (el.style.display = isAuthenticated ? "none" : "block"))
    document.querySelectorAll(".authorized").forEach((el) => (el.style.display = isAuthenticated ? "block" : "none"))

    return isAuthenticated
  }

  if (isAuthenticated) {
    try {
      const accessToken = sessionStorage.getItem("access_token_lightsb")
      const response = await fetch("/api/user", {
        method: "GET",
        credentials: "include",
        headers: {
          Authorization: `Bearer ${accessToken}`,
        },
      })

      if (!response.ok) throw new Error("Failed to load user")

      const data = await response.json()

      // Insert the username into the profile text elements
      document.querySelectorAll(".Profile-txt").forEach((el) => (el.textContent = data.username))
    } catch (error) {
      console.error("Error loading profile:", error)
    }
  }

  // Function to load pages
  async function loadPage(page) {
    try {
      const response = await fetch(`/pages/${page}`)
      if (response.ok) {
        contentArea.innerHTML = await response.text()
        // Initialize page-specific handlers after the content loads
        requestAnimationFrame(() => initPage(page))
      } else {
        contentArea.innerHTML = "<h2>Page not found</h2>"
      }
    } catch (error) {
      contentArea.innerHTML = "<h2>Error loading page</h2>"
    }
  }

  // Function to activate a menu item
  function activateMenuItem(page) {
    menuItems.forEach((i) => i.classList.remove("active"))
    const targetItem = document.querySelector(`.Menu-item[data-page="${page}"]`)
    if (targetItem) targetItem.classList.add("active")
  }

  // Function to check for redirect parameters
  async function checkRedirect() {
    const params = new URLSearchParams(window.location.search)
    if (params.get("redirect") === "profile") {
      await loadPage("profile")
      activateMenuItem("profile")
      history.replaceState({}, "", window.location.pathname)
      return true
    }
    return false
  }

  // Load home page or redirect page
  if (!(await checkRedirect())) {
    await loadPage("home")
    activateMenuItem("home")
  }

  // Menu item click handler
  menuItems.forEach((item) => {
    item.addEventListener("click", async function () {
      const page = this.dataset.page
      activateMenuItem(page)
      await loadPage(page)
    })
  })

  // Initialize page-specific functionality
  function initPage(page) {
    if (page === "profile") {
      initProfilePage() // Profile page initialization
    } else if (page === "profile-change") {
      initProfileEditPage() // Profile edit page initialization
    } else if (page === "generator") {
      initGeneratorPage() // Generator page initialization
    }
  }

  function formatTime(seconds) {
    if (!seconds) {
      return `-`
    } else if (seconds < 1) {
      return `${Math.round(seconds * 1000)} ms`
    } else if (seconds < 60) {
      return `${seconds.toFixed(2)} s`
    } else {
      const minutes = Math.floor(seconds / 60)
      const sec = (seconds % 60).toFixed(0)
      return `${minutes}m ${sec}s`
    }
  }

  function formatDate(dateString) {
    if (!dateString) return "Never used"

    const date = new Date(dateString)
    return date.toLocaleString("en-GB", {
      day: "2-digit",
      month: "short",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    })
  }

  async function initProfilePage() {
    const isAuthenticated = await updateAuthUI()

    const logoutBtn = document.querySelector(".logout-btn")
    const modal = document.getElementById("logout-modal")
    if (modal) modal.style.display = "none"

    const confirmLogout = document.getElementById("confirm-logout")
    const cancelLogout = document.getElementById("cancel-logout")

    const editButton = document.querySelector(".edit-profile-btn")

    if (isAuthenticated) {
      try {
        const accessToken = sessionStorage.getItem("access_token_lightsb")
        const response = await fetch("/api/profile", {
          method: "GET",
          credentials: "include",
          headers: {
            Authorization: `Bearer ${accessToken}`,
          },
        })

        if (!response.ok) throw new Error("Failed to load profile")

        const data = await response.json()

        // Insert profile data into the appropriate page elements
        document.querySelectorAll(".Info-item .Info-value")[0].textContent = data.full_name
        document.querySelectorAll(".Info-item .Info-value")[1].textContent = data.position
        document.querySelectorAll(".Info-item .Info-value")[2].textContent = data.date_of_birth
      } catch (error) {
        console.error("Error loading profile:", error)
      }
    }

    if (isAuthenticated) {
      try {
        const accessToken = sessionStorage.getItem("access_token_lightsb")
        const response = await fetch("/api/stats", {
          method: "GET",
          credentials: "include",
          headers: {
            Authorization: `Bearer ${accessToken}`,
          },
        })

        if (!response.ok) throw new Error("Failed to load stats")

        const data = await response.json()

        document.querySelectorAll(".Stat-item .Stat-value")[0].textContent = data.usage_count
        document.querySelectorAll(".Stat-item .Stat-value")[1].textContent = data.images_count
        document.querySelectorAll(".Stat-item .Stat-value")[2].textContent = formatDate(data.last_used)
        document.querySelectorAll(".Stat-item .Stat-value")[3].textContent = formatTime(data.avg_usage_time / 1000)
      } catch (error) {
        console.error("Error loading stats:", error)
      }
    }

    if (editButton) {
      editButton.addEventListener("click", async (e) => {
        e.preventDefault()
        await loadPage("profile-change")
      })
    }

    // Open logout modal
    logoutBtn.addEventListener("click", (event) => {
      event.preventDefault()
      modal.style.display = "flex"
    })

    // Close modal on cancel
    cancelLogout.addEventListener("click", () => {
      modal.style.display = "none"
    })

    // Confirm logout action
    confirmLogout.addEventListener("click", async () => {
      modal.style.display = "none"

      try {
        logout()
        window.location.reload()
      } catch (error) {
        console.error("Ошибка выхода:", error)
      }
    })

    // Close modal when clicking outside of it
    window.addEventListener("click", (event) => {
      if (event.target === modal) {
        modal.style.display = "none"
      }
    })
  }

  // Profile edit page initialization
  async function initProfileEditPage() {
    const isAuthenticated = await updateAuthUI()

    if (!isAuthenticated) {
      alert("You are not logged in, please log in.")
      window.location.href = "/pages/login"
    }

    const form = document.querySelector(".profile-edit-form")
    if (!form) return

    if (isAuthenticated) {
      try {
        const accessToken = sessionStorage.getItem("access_token_lightsb")
        const response = await fetch("/api/profile", {
          method: "GET",
          credentials: "include",
          headers: {
            Authorization: `Bearer ${accessToken}`,
          },
        })

        if (!response.ok) throw new Error("Failed to load profile")

        const data = await response.json()

        // Populate form fields with profile data
        document.getElementById("full_name").value = data.full_name
        document.getElementById("position").value = data.position
        document.getElementById("date_of_birth").value = data.date_of_birth
      } catch (error) {
        console.error("Error loading profile:", error)
      }
    }

    form.addEventListener("submit", async (e) => {
      e.preventDefault()
      const formData = new FormData(form)
      const formObject = {}
      formData.forEach((value, key) => {
        formObject[key] = value.trim()
      })

      const fullName = formObject.full_name
      const fullNameRegex = /^[А-ЯЁA-Z][а-яёa-z]+\s[А-ЯЁA-Z][а-яёa-z]+(?:\s[А-ЯЁA-Z][а-яёa-z]+)?$/
      if (!fullNameRegex.test(fullName)) {
        alert("Enter correct full name (First and Last name is required, first letter in uppercase)")
        return
      }

      async function sendProfileUpdate(token) {
        return fetch("/api/profile/update", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Accept: "application/json",
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify(formObject),
        })
      }

      const accessToken = sessionStorage.getItem("access_token_lightsb")
      let response = await sendProfileUpdate(accessToken)

      if (response.status === 401) {
        // Token expired
        const newAccessToken = await refreshAccessToken()

        if (newAccessToken) {
          // Retry request with new token
          response = await sendProfileUpdate(newAccessToken)
        } else {
          alert("Session expired, please log in again.")
          window.location.href = "/pages/login"
        }
      }

      try {
        const result = await response.json()
        if (response.ok && result.redirect_to) {
          window.location.href = `/?redirect=${result.redirect_to}`
        } else {
          alert(result.detail || "Update failed")
        }
      } catch (error) {
        console.error("Error processing the response:", error)
        alert("Server error. Try again later.")
      }
    })
  }

  async function initGeneratorPage() {
    await updateAuthUI()

    const form = document.getElementById("generator-form")
    const imageInput = document.getElementById("image-upload")
    const modelSelect = document.getElementById("model-select")
    const loadingText = document.getElementById("loading-text")
    const resultHeader = document.querySelector(".result-header")
    const generatedImagesContainer = document.getElementById("generated-images")
    const downloadAllBtn = document.getElementById("download-all-btn")

    // Store image URLs for download functionality
    let generatedImageUrls = []

    // Function to download a single image
    async function downloadImage(url, filename) {
      try {
        const response = await fetch(url)
        const blob = await response.blob()
        const objectUrl = URL.createObjectURL(blob)

        const link = document.createElement("a")
        link.href = objectUrl
        link.download = filename
        document.body.appendChild(link)
        link.click()
        document.body.removeChild(link)

        // Clean up the object URL
        setTimeout(() => URL.revokeObjectURL(objectUrl), 100)
      } catch (error) {
        console.error("Error downloading image:", error)
        alert("Failed to download image. Please try again.")
      }
    }

    // Function to download all images as a ZIP archive
    async function downloadAllImages() {
      if (generatedImageUrls.length === 0) {
        alert("No images to download.")
        return
      }

      // Add a visual feedback that download is in progress
      downloadAllBtn.textContent = "Creating ZIP..."
      downloadAllBtn.disabled = true

      try {
        // Dynamically import JSZip
        const JSZip = (await import("https://cdn.jsdelivr.net/npm/jszip@3.10.1/+esm")).default
        const zip = new JSZip()

        // Create a folder in the zip
        const imgFolder = zip.folder("generated-images")

        // Add all images to the zip file
        const fetchPromises = generatedImageUrls.map(async (url, index) => {
          const response = await fetch(url)
          const blob = await response.blob()
          imgFolder.file(`generated-image-${index + 1}.png`, blob)
        })

        // Wait for all images to be fetched and added to the zip
        await Promise.all(fetchPromises)

        // Generate the zip file
        const zipBlob = await zip.generateAsync({ type: "blob" })

        // Create a download link for the zip file
        const zipUrl = URL.createObjectURL(zipBlob)
        const link = document.createElement("a")
        link.href = zipUrl
        link.download = "generated-images.zip"
        document.body.appendChild(link)
        link.click()
        document.body.removeChild(link)

        // Clean up the object URL
        setTimeout(() => URL.revokeObjectURL(zipUrl), 100)

        // Reset button state
        downloadAllBtn.innerHTML = '<span class="download-icon">↓</span> Download All'
        downloadAllBtn.disabled = false
      } catch (error) {
        console.error("Error creating ZIP archive:", error)
        downloadAllBtn.innerHTML = '<span class="download-icon">↓</span> Download All'
        downloadAllBtn.disabled = false
        alert("An error occurred while creating the ZIP file. Please try again.")
      }
    }


    // Add click event listener to download button
    if (downloadAllBtn) {
      downloadAllBtn.addEventListener("click", downloadAllImages)
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault()

      const selectedModel = modelSelect.value;

      if (!imageInput.files.length) {
        alert("Please upload an image first.")
        return
      }

      if (!selectedModel) {
        alert("Please select a model first.")
        return
      }

      // Reset previous results
      generatedImageUrls = []

      // Show loading text and hide previous results
      loadingText.style.display = "block"
      resultHeader.style.display = "none"
      generatedImagesContainer.style.display = "none"
      generatedImagesContainer.innerHTML = ""

      const timeStartUse = new Date().toISOString()
      let requestDurationSeconds = 0

      const formData = new FormData()
      formData.append("file", imageInput.files[0])
      formData.append("model", selectedModel)

      console.log("selected model is",selectedModel)

      let f = 0
      let len = 0

      try {
        const startTime = performance.now()

        const response = await fetch("/api/generate", {
          method: "POST",
          body: formData,
        })

        const endTime = performance.now()
        requestDurationSeconds = endTime - startTime
        console.log("Result:", requestDurationSeconds)

        if (!response.ok) {
          throw new Error(`Server responded with ${response.status}: ${response.statusText}`)
        }

        const result = await response.json()

        len = result.image_urls.length

        // Hide loading text
        loadingText.style.display = "none"

        if (result.image_urls.length === 0) {
          alert("No heads were detected in the given image.")
        } else {
          // Store image URLs for download functionality
          generatedImageUrls = [...result.image_urls]

          // Show result header and images container
          resultHeader.style.display = "flex"
          generatedImagesContainer.style.display = "grid"

          const timestamp = Date.now()

          // Apply minimalistic styling to the generated images
          result.image_urls.forEach((url, index) => {
            const imgContainer = document.createElement("div")
            imgContainer.classList.add("image-container")

            const img = document.createElement("img")
            img.src = `${url}?t=${timestamp}`
            img.alt = `Generated Image ${index + 1}`
            img.classList.add("generated-image")

            // Add loading animation
            img.style.opacity = "0"
            img.onload = () => {
              // Fade in effect when image loads
              setTimeout(() => {
                img.style.transition = "opacity 0.5s ease"
                img.style.opacity = "1"
              }, index * 100) // Stagger the animations
            }

            imgContainer.appendChild(img)
            generatedImagesContainer.appendChild(imgContainer)
          })
          f = 1
        }
      } catch (error) {
        console.error("Error during image processing:", error)
        loadingText.style.display = "none"
        resultHeader.style.display = "none"
        alert("An error occurred while processing your image. Please try again.")
      }

      if (f) {
        const formObject = {}
        formObject.usage_count = 1
        formObject.images_count = len
        formObject.last_used = timeStartUse
        formObject.avg_usage_time = requestDurationSeconds

        async function sendStatsUpdate(token) {
          return fetch("/api/stats/update", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Accept: "application/json",
              Authorization: `Bearer ${token}`,
            },
            body: JSON.stringify(formObject),
          })
        }

        const accessToken = sessionStorage.getItem("access_token_lightsb")
        let response = await sendStatsUpdate(accessToken)

        if (response.status === 401) {
          // Token expired
          const newAccessToken = await refreshAccessToken()

          if (newAccessToken) {
            // Retry request with new token
            response = await sendStatsUpdate(newAccessToken)
          } else {
            alert("Session expired, please log in again.")
            window.location.href = "/pages/login"
          }
        }

        try {
          const result = await response.json()
          if (!response.ok) {
            alert(result.detail || "Update failed")
          }
        } catch (error) {
          console.error("Error processing the response:", error)
          alert("Server error. Try again later.")
        }
      }
    })
  }
})
