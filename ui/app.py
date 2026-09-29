"""Dynamic Food Delivery Menu & Allergen RAG - Professional Streamlit Web UI.

Provides a modern food-delivery interface for culinary search and menu recommendations,
enforcing multi-stage query constraints, hybrid dense + BM25 retrieval, cross-encoder reranking,
deterministic allergen safety verification, and grounded citation display.
"""

import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional
import requests
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_API_URL = "https://dynamic-food-delivery-allergen-rag-production.up.railway.app"
ASSET_FOOD_DIR = Path("ui/assets/food")

# Curated food photo mapping with local SVG fallbacks
CATEGORY_IMAGE_URLS = {
    "chicken": "https://images.unsplash.com/photo-1604382354936-07c5d9983bd3?w=600&auto=format&fit=crop&q=80",
    "butter chicken": "https://images.unsplash.com/photo-1588166524941-3bf61a9c41db?w=600&auto=format&fit=crop&q=80",
    "karahi": "https://images.unsplash.com/photo-1603894584373-5ac82b2ae398?w=600&auto=format&fit=crop&q=80",
    "tikka": "https://images.unsplash.com/photo-1599488615731-7e5c2823ff28?w=600&auto=format&fit=crop&q=80",
    "biryani": "https://images.unsplash.com/photo-1563379091339-03b21ab4a4f8?w=600&auto=format&fit=crop&q=80",
    "beef": "https://images.unsplash.com/photo-1558030006-450675393462?w=600&auto=format&fit=crop&q=80",
    "steak": "https://images.unsplash.com/photo-1544025162-d76694265947?w=600&auto=format&fit=crop&q=80",
    "fast food": "https://images.unsplash.com/photo-1568901346375-23c9450c58cd?w=600&auto=format&fit=crop&q=80",
    "burger": "https://images.unsplash.com/photo-1568901346375-23c9450c58cd?w=600&auto=format&fit=crop&q=80",
    "pizza": "https://images.unsplash.com/photo-1513104890138-7c749659a591?w=600&auto=format&fit=crop&q=80",
    "vegetarian": "https://images.unsplash.com/photo-1546069901-ba9599a7e63c?w=600&auto=format&fit=crop&q=80",
    "daal": "https://images.unsplash.com/photo-1546833999-b9f581a1996d?w=600&auto=format&fit=crop&q=80",
    "seafood": "https://images.unsplash.com/photo-1559737558-245bcb566675?w=600&auto=format&fit=crop&q=80",
    "prawns": "https://images.unsplash.com/photo-1559737558-245bcb566675?w=600&auto=format&fit=crop&q=80",
    "dessert": "https://images.unsplash.com/photo-1587314168485-3236d6710814?w=600&auto=format&fit=crop&q=80",
    "baklava": "https://images.unsplash.com/photo-1519676867240-f03562e64548?w=600&auto=format&fit=crop&q=80",
    "bbq": "https://images.unsplash.com/photo-1555939594-58d7cb561ad1?w=600&auto=format&fit=crop&q=80",
    "rice": "https://images.unsplash.com/photo-1512058564366-18510be2db19?w=600&auto=format&fit=crop&q=80",
    "bread": "https://images.unsplash.com/photo-1509440159596-0249088772ff?w=600&auto=format&fit=crop&q=80",
    "soup": "https://images.unsplash.com/photo-1547592166-23ac45744acd?w=600&auto=format&fit=crop&q=80",
    "beverage": "https://images.unsplash.com/photo-1513558161293-cdaf765ed2fd?w=600&auto=format&fit=crop&q=80",
}


def get_dish_image_path_or_url(dish: Dict[str, Any]) -> str:
    """Return an appetizing image URL or reliable local SVG fallback for a dish.

    Visual presentation ONLY; strictly never affects retrieval or ranking.
    """
    dish_name = str(dish.get("dish_name", "")).lower()
    category = str(dish.get("category", "")).lower().replace(" ", "_")

    # Keyword specific match first
    for kw, url in CATEGORY_IMAGE_URLS.items():
        if kw in dish_name:
            return url

    # Category match second
    if category in CATEGORY_IMAGE_URLS:
        return CATEGORY_IMAGE_URLS[category]

    # Local SVG fallback
    local_svg = ASSET_FOOD_DIR / f"{category}.svg"
    if local_svg.is_file():
        return str(local_svg)

    default_svg = ASSET_FOOD_DIR / "default.svg"
    if default_svg.is_file():
        return str(default_svg)

    return "https://images.unsplash.com/photo-1546069901-ba9599a7e63c?w=600&auto=format&fit=crop&q=80"


# =====================================================================
# Backend Communication Layer
# =====================================================================

def query_backend_api(
    payload: Any,
    api_url: str = DEFAULT_API_URL,
) -> Dict[str, Any]:
    """Execute a query against the FastAPI backend, with graceful fallback to in-process client.

    Args:
        payload: Dict matching QueryRequest schema or raw query string.
        api_url: Base HTTP URL for the FastAPI service.

    Returns:
        Structured response dictionary matching QueryResponse schema.
    """
    if isinstance(payload, str):
        payload = {"query": payload}
    clean_url = api_url.rstrip("/")
    try:
        resp = requests.post(f"{clean_url}/query", json=payload, timeout=60)
            return resp.json()
        elif resp.status_code == 422:
            return {
                "status": "INVALID_INPUT",
                "query": payload.get("query", ""),
                "answer": "Input validation error: Please verify your query parameters.",
                "warnings": [f"HTTP 422: {resp.text}"],
                "sources": [],
                "citations": [],
                "results": [],
            }
        else:
            return {
                "status": "ERROR",
                "query": payload.get("query", ""),
                "answer": f"Backend API returned status code {resp.status_code}.",
                "warnings": [resp.text],
                "sources": [],
                "citations": [],
                "results": [],
            }
    except Exception as e:
        logger.info("Direct HTTP connection failed (%s); utilizing in-process pipeline client.", e)
        try:
            from pydantic import ValidationError
            from api.main import execute_query
            from api.schemas import QueryRequest

            try:
                req = QueryRequest(**payload)
            except (ValueError, ValidationError) as val_err:
                return {
                    "status": "INVALID_INPUT",
                    "query": payload.get("query", ""),
                    "answer": "Input validation error: Please verify your query parameters.",
                    "warnings": [f"HTTP 422: {str(val_err)}"],
                    "sources": [],
                    "citations": [],
                    "results": [],
                }

            res = execute_query(req)
            data = res.model_dump()
            data["_in_process_fallback"] = True
            return data
        except Exception as inner_e:
            logger.error("Failed in-process execution fallback: %s", inner_e)
            return {
                "status": "ERROR",
                "query": payload.get("query", ""),
                "answer": f"In-process engine error: {str(inner_e)}.",
                "warnings": [str(inner_e)],
                "sources": [],
                "citations": [],
                "results": [],
            }


def fetch_health_status(api_url: str = DEFAULT_API_URL) -> Dict[str, Any]:
    """Check health and readiness of the backend services."""
    clean_url = api_url.rstrip("/")
    try:
        resp = requests.get(f"{clean_url}/health", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass

    try:
        from api.main import get_health

        return get_health().model_dump()
    except Exception as e:
        return {"status": "unreachable", "error": str(e)}


# =====================================================================
# UI Presentation Components
# =====================================================================

def inject_custom_styles() -> None:
    """Inject polished, modern CSS styling for a food-delivery platform."""
    st.markdown(
        """
        <style>
        /* Base typography & layout smoothing */
        html, body, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }

        /* Hero banner container */
        .hero-banner {
            background: linear-gradient(135deg, #1A1F2C 0%, #2D3748 100%);
            border-radius: 16px;
            padding: 32px 36px;
            color: #FFFFFF;
            margin-bottom: 24px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.12);
            border: 1px solid rgba(255,255,255,0.08);
        }
        .hero-title {
            font-size: 2.2rem;
            font-weight: 800;
            color: #FFFFFF;
            margin-bottom: 8px;
            letter-spacing: -0.5px;
        }
        .hero-subtitle {
            font-size: 1.05rem;
            color: #E2E8F0;
            line-height: 1.5;
            max-width: 720px;
        }
        .hero-pill-badge {
            display: inline-block;
            background: rgba(226, 55, 68, 0.2);
            border: 1px solid rgba(226, 55, 68, 0.4);
            color: #FF6B6B;
            font-size: 0.8rem;
            font-weight: 700;
            padding: 4px 12px;
            border-radius: 20px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 12px;
        }

        /* Dish Card Styling */
        .dish-card-wrapper {
            background: #FFFFFF;
            border-radius: 16px;
            border: 1px solid #E2E8F0;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.04);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .dish-card-wrapper:hover {
            box-shadow: 0 8px 22px rgba(0,0,0,0.08);
            border-color: #CBD5E1;
        }

        /* Pill badges */
        .badge-pill {
            display: inline-flex;
            align-items: center;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-right: 6px;
            margin-bottom: 4px;
        }
        .badge-halal {
            background-color: #DEF7EC;
            color: #03543F;
            border: 1px solid #BCF0DA;
        }
        .badge-nonhalal {
            background-color: #FDE8E8;
            color: #9B1C1C;
            border: 1px solid #FBD5D5;
        }
        .badge-price {
            background-color: #FEF08A;
            color: #854D0E;
            font-weight: 800;
            font-size: 0.95rem;
            padding: 4px 12px;
            border-radius: 8px;
        }
        .badge-spice {
            background-color: #FFEDD5;
            color: #C2410C;
            border: 1px solid #FED7AA;
        }
        .badge-loc {
            background-color: #F1F5F9;
            color: #334155;
            border: 1px solid #E2E8F0;
        }
        .badge-evidence {
            background-color: #EEF2FF;
            color: #3730A3;
            border: 1px solid #C7D2FE;
            font-family: monospace;
        }

        /* Allergen status boxes */
        .allergen-box-verified {
            background-color: #F0FDF4;
            border-left: 4px solid #22C55E;
            padding: 10px 14px;
            border-radius: 6px;
            color: #15803D;
            font-size: 0.88rem;
            font-weight: 600;
            margin-top: 8px;
        }
        .allergen-box-unknown {
            background-color: #FFFBEB;
            border-left: 4px solid #F59E0B;
            padding: 10px 14px;
            border-radius: 6px;
            color: #B45309;
            font-size: 0.88rem;
            font-weight: 600;
            margin-top: 8px;
        }
        .allergen-box-conflict {
            background-color: #FEF2F2;
            border-left: 4px solid #EF4444;
            padding: 10px 14px;
            border-radius: 6px;
            color: #B91C1C;
            font-size: 0.88rem;
            font-weight: 600;
            margin-top: 8px;
        }
        .allergen-box-declared {
            background-color: #EFF6FF;
            border-left: 4px solid #3B82F6;
            padding: 10px 14px;
            border-radius: 6px;
            color: #1D4ED8;
            font-size: 0.88rem;
            font-weight: 600;
            margin-top: 8px;
        }

        /* Advisory banner */
        .advisory-banner {
            background-color: #FEF2F2;
            border: 1px solid #FCA5A5;
            border-radius: 12px;
            padding: 16px 20px;
            color: #991B1B;
            margin-bottom: 20px;
        }

        /* Footer styling */
        .app-footer {
            margin-top: 48px;
            padding: 24px 0;
            text-align: center;
            border-top: 1px solid #E2E8F0;
            color: #64748B;
            font-size: 0.88rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_dish_card(dish: Dict[str, Any], idx: int) -> None:
    """Render a single structured menu item card with complete safety metadata."""
    dish_name = dish.get("dish_name", "Unknown Dish")
    restaurant = dish.get("restaurant_name", "Unknown Restaurant")
    price = float(dish.get("price_pkr", 0.0))
    location = dish.get("location", "Unknown Location")
    halal = dish.get("halal", True)
    spice = dish.get("spice_level", "Medium")
    allergens = dish.get("allergens", [])
    allergen_status = dish.get("allergen_status", "UNKNOWN")
    verification_status = dish.get("verification_status")
    source_id = dish.get("source_id", f"dish_{idx}")
    source_menu = dish.get("source", "Standard Menu")
    match_reason = dish.get("match_reason", "")
    warning = dish.get("warning")
    image_url_or_path = get_dish_image_path_or_url(dish)

    with st.container():
        # Clean 2-column card layout: image on left, rich metadata on right
        card_col1, card_col2 = st.columns([1, 2.6])

        with card_col1:
            try:
                st.image(image_url_or_path, use_container_width=True)
            except Exception:
                fallback_svg = ASSET_FOOD_DIR / "default.svg"
                if fallback_svg.is_file():
                    st.image(str(fallback_svg), use_container_width=True)
                else:
                    st.image("https://images.unsplash.com/photo-1546069901-ba9599a7e63c?w=600&auto=format&fit=crop&q=80", use_container_width=True)

        with card_col2:
            st.markdown(f"### {idx}. {dish_name} `[{source_id}]`")
            
            # Badge row
            b_cols = st.columns([1.5, 1, 1, 1])
            b_cols[0].write(f"🏢 **{restaurant}**")
            b_cols[1].markdown(f"<span class='badge-price'>PKR {price:,.0f}</span>", unsafe_allow_html=True)
            b_cols[2].write(f"📍 **{location}**")
            b_cols[3].write("🟢 **Halal**" if halal else "🔴 **Non-Halal**")

            st.write(f"🌶️ **Spice Level:** {spice}")

            # Visual Allergen Safety Callouts
            if allergen_status == "KNOWN_NO_ALLERGENS":
                st.markdown(
                    "<div class='allergen-box-verified'>🛡️ <b>Allergen Status:</b> Verified Zero Listed Allergens (Safe)</div>",
                    unsafe_allow_html=True,
                )
            elif allergen_status == "UNKNOWN" or verification_status == "INSUFFICIENT_EVIDENCE":
                st.markdown(
                    "<div class='allergen-box-unknown'>⚠️ <b>Allergen Status:</b> Unverified / Unknown Data — Cannot guarantee allergen-free status.</div>",
                    unsafe_allow_html=True,
                )
            elif allergen_status == "CONFLICT" or verification_status == "CONFLICTING_EVIDENCE":
                st.markdown(
                    "<div class='allergen-box-conflict'>🚨 <b>Allergen Status:</b> Conflicting Evidence Detected — Cannot guarantee allergen-free status.</div>",
                    unsafe_allow_html=True,
                )
            elif allergens:
                st.markdown(
                    f"<div class='allergen-box-declared'>⚠️ <b>Contains Declared Allergens:</b> {', '.join(allergens)}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.write("Allergens: Unspecified")

            st.caption(f"📖 **Evidence Source:** {source_menu} `[{source_id}]`")

            if match_reason:
                st.info(f"**Why it matches:** {match_reason}")

            if warning:
                st.warning(f"**Safety Advisory:** {warning}")

        st.divider()


def main():
    """Main Streamlit execution entry point."""
    st.set_page_config(
        page_title="Dynamic Food Delivery Menu & Allergen RAG",
        page_icon="🍲",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_custom_styles()

    # Title contract preserved for AppTest
    st.title("🍲 Dynamic Food Delivery Menu & Allergen RAG")

    # 1. Hero Section
    st.markdown(
        """
        <div class="hero-banner">
            <span class="hero-pill-badge">✨ Grounded Culinary Intelligence</span>
            <div class="hero-title">Find the perfect meal</div>
            <div class="hero-subtitle">
                Explore verified menus from leading Pakistani restaurants & cloud kitchens with deterministic 
                allergen safety verification, numerical budget filtering, and grounded citation proofs.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Sidebar: Backend Configuration and Hard Constraints
    st.sidebar.markdown("### ⚙️ Delivery Filters")
    api_url = st.sidebar.text_input("Backend API Endpoint", value=DEFAULT_API_URL)

    # Health check pill
    health_status = fetch_health_status(api_url)
    if health_status.get("status") == "healthy":
        st.sidebar.success(f"🟢 **API Online** ({health_status.get('total_indexed_items', 35)} Dishes)")
    else:
        st.sidebar.warning("🟠 **Offline Fallback** (Using In-Process RAG)")

    with st.sidebar.expander("🔍 System Readiness Diagnostics", expanded=False):
        if st.button("Probe Health Status"):
            st.json(health_status)

    st.sidebar.subheader("🎯 Structured Constraint Overrides")

    location_choice = st.sidebar.selectbox(
        "Delivery Location / Zone",
        options=["Any Location", "Gulberg", "DHA", "Mall Road", "F-7", "Blue Area", "Saddar", "Clifton"],
        index=0,
    )

    halal_only = st.sidebar.checkbox("Halal Certified Only", value=True)

    max_price = st.sidebar.slider(
        "Maximum Budget (PKR)",
        min_value=0,
        max_value=4000,
        value=0,
        step=100,
        help="Set to 0 for no price ceiling.",
    )

    spice_preference = st.sidebar.selectbox(
        "Spice Level Preference",
        options=["Any", "Mild", "Medium", "Spicy"],
        index=0,
    )

    excluded_allergens = st.sidebar.multiselect(
        "Strict Allergen Exclusions",
        options=["dairy", "peanuts", "tree_nuts", "gluten", "soy", "eggs", "shellfish", "fish", "sesame"],
        default=[],
        help="Dishes declaring or containing these ingredients will be excluded deterministically.",
    )

    top_k = st.sidebar.slider("Max Recommendations (Top-K)", min_value=1, max_value=10, value=5)

    # 3. Search Box & Quick Prompts
    col1, col2 = st.columns([4, 1])
    with col1:
        query_input = st.text_input(
            "Enter your food request:",
            placeholder="e.g. Find high-protein Halal chicken meals under PKR 1500 without dairy in Gulberg",
            key="user_query_input",
        )
    with col2:
        st.write("")
        st.write("")
        search_clicked = st.button("🔎 Search Menu", type="primary", use_container_width=True)

    # Quick Prompts
    st.markdown("**Popular Recommendations & Safety Tests:**")
    example_cols = st.columns(4)
    if example_cols[0].button("🍗 Halal Chicken in Gulberg"):
        query_input = "Halal chicken under 1500 without dairy in Gulberg"
        search_clicked = True
    if example_cols[1].button("🍔 Gluten-free DHA Burger"):
        query_input = "Gluten-free beef burger under 2000 in DHA"
        search_clicked = True
    if example_cols[2].button("⚠️ Mystery Daal (UNKNOWN)"):
        query_input = "Mystery Special Daily Daal"
        search_clicked = True
    if example_cols[3].button("🚨 Secret Karahi (CONFLICT)"):
        query_input = "Chefs Secret Karahi"
        search_clicked = True

    # 4. Search Execution & Results
    if search_clicked and query_input:
        payload: Dict[str, Any] = {
            "query": query_input.strip(),
            "top_k": top_k,
        }
        if location_choice != "Any Location":
            payload["location"] = location_choice
        if halal_only:
            payload["halal"] = True
        if max_price > 0:
            payload["max_price_pkr"] = float(max_price)
        if spice_preference != "Any":
            payload["spice_level"] = spice_preference
        if excluded_allergens:
            payload["excluded_allergens"] = excluded_allergens

        with st.spinner("Executing grounded hybrid retrieval, cross-encoder reranking, and safety verification..."):
            response = query_backend_api(payload, api_url=api_url)

        status_code = response.get("status", "UNKNOWN")

        # Status Banner Presentation
        st.write("---")
        if status_code == "SUPPORTED":
            st.success("### ✅ Status: SUPPORTED — Verified Menu Recommendations")
        elif status_code == "PARTIALLY_SUPPORTED":
            st.info("### ℹ️ Status: PARTIALLY_SUPPORTED — General Match with Unverified Allergen Records")
        elif status_code == "INSUFFICIENT_EVIDENCE":
            st.warning("### ⚠️ Status: INSUFFICIENT_EVIDENCE — Allergen Information Unverified")
        elif status_code == "CONFLICTING_EVIDENCE":
            st.error("### 🚨 Status: CONFLICTING_EVIDENCE — Contradictory Allergen Evidence Detected")
        elif status_code == "NO_MATCHING_ITEMS":
            st.warning("### 🚫 Status: NO_MATCHING_ITEMS — No Dishes Satisfy All Constraints")
        elif status_code == "INVALID_INPUT":
            st.error("### ⚠️ Status: INVALID_INPUT — Please check your query parameters.")
        else:
            st.error(f"### ❌ Status: {status_code}")

        # Active Allergen Safety Advisories Callout
        warnings = response.get("warnings", [])
        if warnings:
            with st.container():
                st.markdown(
                    "<div class='advisory-banner'><h4>⚠️ Active Allergen Safety Advisories</h4>",
                    unsafe_allow_html=True,
                )
                for w in warnings:
                    st.write(f"- {w}")
                st.markdown("</div>", unsafe_allow_html=True)

        # Grounded Answer Summary Expander
        answer_text = response.get("answer", "")
        if answer_text:
            with st.expander("📝 Grounded Synthesis Response", expanded=True):
                st.markdown(answer_text)

        # Recommended Dishes Card Grid
        results = response.get("results", [])
        if results:
            st.subheader(f"🍴 Recommended Dishes ({len(results)} Found)")
            for idx, dish in enumerate(results, start=1):
                render_dish_card(dish, idx)
        elif status_code == "NO_MATCHING_ITEMS":
            st.info("Try relaxing your budget, removing allergen exclusions, or changing delivery zones.")

        # Structured Evidence Citations & Telemetry Area
        citations = response.get("citations", [])
        latency = response.get("latency", {})
        constraints = response.get("parsed_constraints", {})

        with st.expander("📊 Evidence, Citations & Grounding Audit Telemetry", expanded=False):
            t_col1, t_col2 = st.columns(2)
            with t_col1:
                st.markdown("**Structured Verified Citations:**")
                if citations:
                    for cit in citations:
                        st.markdown(
                            f"- **`[{cit.get('source_id')}]` {cit.get('dish_name')}** ({cit.get('restaurant_name')}) | "
                            f"PKR {cit.get('price_pkr', 0):,.0f} | Status: `{cit.get('allergen_status')}`"
                        )
                else:
                    st.write("No citations available.")

            with t_col2:
                st.markdown("**Pipeline Execution Latencies (ms):**")
                st.json(latency)
                st.markdown("**Parsed Constraints & Filters:**")
                st.json(constraints)

    # 5. Professional Footer
    st.markdown(
        """
        <div class="app-footer">
            <b>Food Delivery Menu RAG</b> • Powered by RAG • Grounded responses • Allergen-aware recommendations<br>
            <span style="font-size: 0.8rem; color: #94A3B8;">
                Deterministic verification gate • SentenceTransformers & CrossEncoder • ChromaDB & Okapi BM25
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
