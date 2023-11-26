import streamlit as st
from streamlit_drawable_canvas import st_canvas
from utils.gaugan import api_prediction
from utils.image_processing import ELEMENTS, ELEMENTS_2_COLOR, convert_img_for_gaugan



def main():

    st.markdown(
    """
    This AI application harnesses the capabilities of a GAN to produce artistry reminiscent of Bob Ross. Operating on segmentation masks, it meticulously crafts paintings that faithfully embody the distinctive style associated with the renowned artist. 

    View the source code on [github](https://github.com/szymciem8/Semantic-Image-Synthesis).
    
    """
    )

    st.sidebar.subheader("Settings")
    drawing_element = st.sidebar.selectbox(
        "Drawing element:",
        [el for el in ELEMENTS.values()]
    )
    stroke_width = st.sidebar.slider("Stroke width: ", 3, 50, 20)
    realtime_update = st.sidebar.checkbox("Update in realtime", True)

    prediction = None
    with st.form('Painter'):
        col1, col2 = st.columns(2)
        with col1:
            canvas_result = st_canvas(
                stroke_width=stroke_width,
                stroke_color=ELEMENTS_2_COLOR[drawing_element],
                background_color=ELEMENTS_2_COLOR['sky'],
                background_image=None,
                update_streamlit=realtime_update,
                height=250,
                width=250,
                drawing_mode='freedraw',
                point_display_radius=0,
                display_toolbar=st.sidebar.checkbox("Display toolbar", True),
                key="full_app",
            )
        generate = st.form_submit_button(label="Generate", help="Click to generate or regenerate!")
        if canvas_result.image_data is not None:
            if generate:
                input_image = canvas_result.image_data[:,:,:3]
                image = convert_img_for_gaugan(input_image)
                prediction = api_prediction(image)
                st.session_state['prediction'] = prediction
                with col2:
                    st.image(prediction)
        with col2:
            if prediction is None and st.session_state.get('prediction') is not None:
                st.image(st.session_state['prediction'])


if __name__ == "__main__":
    st.set_page_config(
        page_title="Bob Ross AI Painter", page_icon=":pencil2:"
    )
    st.title("Bob Ross AI Painter")
    st.sidebar.header("Configuration")
    main()