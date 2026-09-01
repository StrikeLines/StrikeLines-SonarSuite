import napari
from magicgui import magicgui
from pathlib import Path
from sidescantools.sidescan_preproc import SidescanPreprocessor
import numpy as np
from sidescantools.sidescan_file import SidescanFile
import os

from sidescantools.interaction_mode import InteractionMode, InteractionModeController
from sidescantools.bottom_line_io import (
    compute_depth_info,
    save_bottom_info,
    load_bottom_info,
)
from sidescantools.contact_picker import (
    ContactPickerService,
    display_position_for_anchor,
)
from sidescantools.contact_store import ContactStore, DuplicateContactAnchor
from sidescantools.contact_thumbnail import ContactThumbnailExtractor
from sidescantools.contact_ui import ContactDock
from sidescantools.georef_thread import Georeferencer
from sidescantools.swath_geometry import GeometrySettings


def run_napari_btm_line(
    filepath: str | os.PathLike,
    chunk_size=1000,
    default_threshold=0.7,
    downsampling_factor=1,
    contrast_limit=0.0,
    work_dir=None,
    active_dB=False,
    active_hist_equal=False,
    contact_pick_callback=None,
    contacts_db_path=None,
    geometry_settings=None,
    block=True,
):
    """Run bottom line detection in napari on a given file

    Parameters
    ----------
    filepath: str | os.PathLike
        Path to sidescan file
    chunk_size: int
        Number of pings per single chunk
    default_threshold: float
        Number in range [0, 1] that is used as threshold for binarization of the image before the edges are detected
    downsampling_factor: int
        Factor used for decimation of ping signals
    work_dir: str | os.PathLike
        Path to desired directory that is used as default directory for saving/loading of results to ``.npz`` files
    active_dB: bool
        If ``True`` data will be converted to dB for display in napari
    contact_pick_callback: callable | None
        Optional persistence callback receiving one rounded data position. If it
        returns a contact record, a marker is added after the callback succeeds.
    """
    filepath = Path(filepath)
    add_line_width = 1  # additional line width for plotting of bottom line
    interaction_modes = InteractionModeController()
    if geometry_settings is None:
        geometry_settings = GeometrySettings(vertical_beam_angle=60)

    sidescan_file = SidescanFile(filepath)
    preproc = SidescanPreprocessor(
        sidescan_file=sidescan_file,
        chunk_size=chunk_size,
        downsampling_factor=downsampling_factor,
    )

    # Init bottom detection by doing an initial guess
    depth_info = compute_depth_info(sidescan_file, downsampling_factor)

    print("Initializing napari UI for Bottom Detection")
    preproc.init_napari_bottom_detect(
        default_threshold,
        active_dB=active_dB,
        active_hist_equal=active_hist_equal,
        depth_info=depth_info,
    )

    # build napari GUI
    @magicgui(
        auto_call=True,
        threshold_bin={
            "widget_type": "FloatSlider",
            "min": 0,
            "max": 1.0,
            "step": 0.01,
        },
        choose_strategy={
            "widget_type": "RadioButtons",
            "choices": preproc.bottom_strategy_choices,
        },
        call_button="Recalculate",
    )
    def widget_thresh(
        viewer: napari.Viewer,
        threshold_bin=default_threshold,
        choose_strategy=preproc.bottom_strategy_choices[1],
    ):

        slider_position = viewer.dims.current_step[0]
        preproc.detect_bottom_napari(
            slider_position,
            threshold_bin=threshold_bin,
            bottom_strategy_choice=choose_strategy,
            add_line_width=add_line_width,
        )

        # update bottom plot with new data
        bottom_image_layer.data = preproc.bottom_map
        # update edge plot
        press_b(viewer)

    # Build widget to load depth data
    call_button_text = "d: Load depth data from file"
    if depth_info is None:
        call_button_text = "No depth data found"

    @magicgui(
        auto_call=True,
        depth_offset={
            "widget_type": "IntSlider",
            "min": -1 * preproc.ping_len,
            "max": preproc.ping_len,
            "step": 1,
        },
        call_button=call_button_text,
    )
    def widget_depth(viewer: napari.Viewer, depth_offset=0):
        preproc.set_depth_from_info(offset=depth_offset)
        # update bottom plot with new data
        bottom_image_layer.data = preproc.bottom_map

    # Build widget for aux parameters that shall not trigger a recalculation for the current chunk
    @magicgui(auto_call=True, call_button=None)
    def manual_annotation_widget(activate_manual_annotation: bool):
        if activate_manual_annotation:
            interaction_modes.set_mode(InteractionMode.BOTTOM_EDIT)
        elif interaction_modes.mode is InteractionMode.BOTTOM_EDIT:
            interaction_modes.set_mode(InteractionMode.PAN_ZOOM)

    @magicgui(auto_call=False, call_button=None)
    def interaction_mode_widget(active_mode: str = "Pan/Zoom"):
        pass

    # Build saving and loading widgets using simple npz file
    if work_dir is None:
        default_bottom_path = filepath.parent / (filepath.stem + "_bottom_info.npz")
    else:
        work_dir = Path(work_dir)
        default_bottom_path = work_dir / (filepath.stem + "_bottom_info.npz")

    @magicgui(filename={"mode": "w"}, call_button="Save")
    def filepicker_save(
        filename=default_bottom_path,
    ):
        save_bottom_info(filename, preproc, sidescan_file)

    @magicgui(filename={"mode": "r"}, call_button="Load")
    def filepicker_load(
        filename=default_bottom_path,
    ):
        if filename.exists() and filename.suffix == ".npz":
            load_bottom_info(filename, preproc, sidescan_file)
            bottom_image_layer.refresh()

    viewer = napari.Viewer(title="SidescanTools - Bottom line detection")

    # add custom shortcuts
    @viewer.bind_key("m")
    def press_m(viewer):
        interaction_modes.toggle(InteractionMode.BOTTOM_EDIT)

    @viewer.bind_key("t")
    def press_t(viewer):
        interaction_modes.toggle(InteractionMode.TARGET_PICK)

    @viewer.bind_key("Escape")
    def press_escape(viewer):
        interaction_modes.set_mode(InteractionMode.PAN_ZOOM)

    @viewer.bind_key("r")
    def press_r(viewer):
        widget_thresh.changed()

    @viewer.bind_key("b")
    def press_b(viewer):
        binarized_image_layer.data = preproc.napari_fullmat_bin
        edges_image_layer.data = preproc.edges_mat

    @viewer.bind_key("d")
    def press_d(viewer):
        widget_depth.changed()

    # add image
    binarized_image_layer = viewer.add_image(
        preproc.napari_fullmat_bin, name="binarized image"
    )
    edges_image_layer = viewer.add_image(preproc.edges_mat, name="edges")
    sidescan_image_layer = viewer.add_image(
        preproc.napari_fullmat, name="sidescan image", colormap="copper"
    )

    # show bottom line overlay
    colors = [[1, 1, 1, 0], [1, 0, 0, 1]]  # r,g,b,alpha
    bottom_colormap = {
        "colors": colors,
        "name": "bottom_line_cmap",
        "interpolation": "linear",
    }
    bottom_image_layer = viewer.add_image(
        preproc.bottom_map, name="bottom_map", colormap=bottom_colormap
    )
    contacts_layer = viewer.add_points(
        np.empty((0, 3)),
        name="Contacts",
        size=12,
        face_color="cyan",
        edge_color="white",
    )

    # Project contact persistence is initialized with the dock. Geometry itself
    # remains lazy and is prepared off the UI thread on first Target Pick use.
    if contacts_db_path is None:
        contact_root = Path(work_dir) if work_dir is not None else filepath.parent
        contacts_db_path = contact_root / "contacts.sqlite"
    contact_store = ContactStore(contacts_db_path)
    source_stat = filepath.stat()
    contact_source = contact_store.register_source_file(
        filepath,
        format=filepath.suffix.lstrip("."),
        ping_count=sidescan_file.num_ping,
        source_sample_count=sidescan_file.ping_len,
        file_size_bytes=source_stat.st_size,
        mtime_ns=source_stat.st_mtime_ns,
    )
    geometry_profile_id = contact_store.get_or_create_geometry_profile(
        geometry_settings
    )
    contact_store.mark_stale_for_profile(contact_source.id, geometry_profile_id)
    contact_dock = ContactDock(
        contact_store,
        contact_source.id,
        export_directory=Path(contacts_db_path).parent,
    )
    viewer.window.add_dock_widget(contact_dock, name="Sonar Contacts", area="right")
    contact_runtime = {"worker": None, "picker": None}

    def rebuild_contact_markers(*args):
        markers = [
            display_position_for_anchor(
                record.draft.anchor,
                chunk_size=preproc.chunk_size,
                display_channel_width=preproc.ping_len,
            )
            for record in contact_store.list_contacts(
                source_file_id=contact_source.id
            )
        ]
        contacts_layer.data = (
            np.asarray(markers, dtype=float) if markers else np.empty((0, 3))
        )

    rebuild_contact_markers()
    contact_dock.contact_deleted.connect(rebuild_contact_markers)

    def start_contact_geometry():
        nonlocal contact_pick_callback
        if contact_runtime["picker"] is not None:
            return
        worker = contact_runtime["worker"]
        if worker is not None and worker.is_running:
            return
        from napari.qt.threading import thread_worker

        @thread_worker
        def prepare_geometry():
            prepared = {}
            for channel in (0, 1):
                georeferencer = Georeferencer(
                    filepath,
                    channel=channel,
                    sidescan_file=sidescan_file,
                    geometry_settings=geometry_settings,
                    output_folder=filepath.parent,
                )
                prepared[channel] = georeferencer.prepare_swath_geometry()
            return prepared

        def geometry_ready(prepared):
            nonlocal contact_pick_callback
            thumbnail_extractor = ContactThumbnailExtractor(
                preprocessor=preproc,
                sidescan_file=sidescan_file,
            )
            picker = ContactPickerService(
                sidescan_file=sidescan_file,
                preprocessor=preproc,
                source_file_id=contact_source.id,
                geometry_profile_id=geometry_profile_id,
                geometry_by_channel=prepared,
                store=contact_store,
                thumbnail_factory=thumbnail_extractor,
            )
            contact_runtime["picker"] = picker
            contact_dock.set_geometry_status("Geometry ready", ready=True)

            def save_contact(position):
                chunk_index, local_ping_index, display_x = position
                try:
                    result = picker.pick_display_pixel(
                        chunk_index=chunk_index,
                        local_ping_index=local_ping_index,
                        display_x=display_x,
                    )
                except DuplicateContactAnchor:
                    viewer.status = "A contact already exists at this sonar sample"
                    return None
                except Exception as exc:
                    viewer.status = f"Contact not saved: {exc}"
                    return None
                contact_dock.refresh_and_focus_name(select_contact_id=result.contact.id)
                viewer.status = (
                    result.thumbnail_warning
                    or f"Saved {result.contact.draft.name}"
                )
                return result

            contact_pick_callback = save_contact

        def geometry_failed(error):
            contact_dock.set_geometry_status(f"Geometry error: {error}")
            viewer.status = f"Contact geometry failed: {error}"

        contact_dock.set_geometry_status("Preparing contact geometry…")
        worker = prepare_geometry()
        contact_runtime["worker"] = worker
        worker.returned.connect(geometry_ready)
        worker.errored.connect(geometry_failed)
        worker.start()

    def prepare_on_target_mode(mode):
        if mode is InteractionMode.TARGET_PICK:
            start_contact_geometry()

    interaction_modes.add_listener(prepare_on_target_mode)

    # add widgets to main window
    viewer.window.add_dock_widget(widget_thresh, name="Bottom detection parameters")
    viewer.window.add_dock_widget(
        widget_depth, name="Define Bottom Line via intern depth data"
    )
    widget_thresh.visible = False  # HACK to change size policy...
    viewer.window.add_dock_widget(
        manual_annotation_widget, name="Activate manual annotation"
    )
    viewer.window.add_dock_widget(interaction_mode_widget, name="Interaction mode")
    viewer.window.add_dock_widget(filepicker_save, name="Save to")
    viewer.window.add_dock_widget(filepicker_load, name="Load from")
    widget_thresh.visible = True
    # label shortcuts
    widget_thresh.threshold_bin.label = "Threshold binarization [0,1]"
    widget_thresh.choose_strategy.label = "Choose strategy"
    widget_thresh.call_button.text = "r: Recalculate"
    widget_depth.depth_offset.label = "Depth Offset in samples"
    if depth_info is None:
        widget_depth.call_button.enabled = False
        widget_depth._auto_call = False
    manual_annotation_widget.activate_manual_annotation.text = (
        "m: Activate manual annotation"
    )
    interaction_mode_widget.active_mode.enabled = False
    filepicker_save.filename.label = "File"
    filepicker_load.filename.label = "File"

    def apply_interaction_mode(mode):
        manual_active = mode is InteractionMode.BOTTOM_EDIT
        if manual_annotation_widget.activate_manual_annotation.value != manual_active:
            manual_annotation_widget.activate_manual_annotation.value = manual_active
        interaction_mode_widget.active_mode.value = mode.label
        for current_layer in viewer.layers:
            current_layer.mouse_pan = mode is InteractionMode.PAN_ZOOM

    interaction_modes.add_listener(apply_interaction_mode)
    apply_interaction_mode(interaction_modes.mode)

    def bottom_edit_mouse_callback(layer, event):
        if (
            interaction_modes.mode is InteractionMode.BOTTOM_EDIT
            and event.button == 1
            and 0 <= np.round(event.position[1]) < layer.data.shape[1]
            and 0 <= np.round(event.position[2]) < layer.data.shape[2]
        ):

            # print('mouse down')
            dragged = False
            yield

            # on move
            last_pos = np.zeros(3)
            while (
                event.type == "mouse_move"
                and 0 <= np.round(event.position[1]) < layer.data.shape[1]
                and 0 <= np.round(event.position[2]) < layer.data.shape[2]
            ):
                # print('mouse move')
                dragged = True

                cur_pos = np.array(np.round(event.position), dtype=int)
                if cur_pos[2] < layer.data.shape[2] / 2:
                    preproc.napari_portside_bottom[cur_pos[0], cur_pos[1]] = cur_pos[2]
                    if (
                        widget_thresh.choose_strategy.value
                        == preproc.bottom_strategy_choices[1]
                    ):
                        preproc.napari_starboard_bottom[cur_pos[0], cur_pos[1]] = (
                            layer.data.shape[2] / 2 - cur_pos[2]
                        )
                else:
                    preproc.napari_starboard_bottom[cur_pos[0], cur_pos[1]] = (
                        cur_pos[2] - layer.data.shape[2] / 2
                    )
                    if (
                        widget_thresh.choose_strategy.value
                        == preproc.bottom_strategy_choices[1]
                    ):
                        preproc.napari_portside_bottom[cur_pos[0], cur_pos[1]] = (
                            layer.data.shape[2] - cur_pos[2]
                        )

                # check whether movement skipped points and do linear interpolation
                if (last_pos[1:] > 0).all():
                    if last_pos[1] - cur_pos[1] > 1:
                        if cur_pos[2] < layer.data.shape[2] / 2:
                            preproc.napari_portside_bottom[
                                cur_pos[0], cur_pos[1] : last_pos[1]
                            ] = cur_pos[2]
                            if (
                                widget_thresh.choose_strategy.value
                                == preproc.bottom_strategy_choices[1]
                            ):
                                preproc.napari_starboard_bottom[
                                    cur_pos[0], cur_pos[1] : last_pos[1]
                                ] = (layer.data.shape[2] / 2 - cur_pos[2])
                        else:
                            preproc.napari_starboard_bottom[
                                cur_pos[0], cur_pos[1] : last_pos[1]
                            ] = (cur_pos[2] - layer.data.shape[2] / 2)
                            if (
                                widget_thresh.choose_strategy.value
                                == preproc.bottom_strategy_choices[1]
                            ):
                                preproc.napari_portside_bottom[
                                    cur_pos[0], cur_pos[1] : last_pos[1]
                                ] = (layer.data.shape[2] - cur_pos[2])
                    elif last_pos[1] - cur_pos[1] < 1:
                        if cur_pos[2] < layer.data.shape[2] / 2:
                            preproc.napari_portside_bottom[
                                cur_pos[0], last_pos[1] : cur_pos[1]
                            ] = cur_pos[2]
                            if (
                                widget_thresh.choose_strategy.value
                                == preproc.bottom_strategy_choices[1]
                            ):
                                preproc.napari_starboard_bottom[
                                    cur_pos[0], last_pos[1] : cur_pos[1]
                                ] = (layer.data.shape[2] / 2 - cur_pos[2])
                        else:
                            preproc.napari_starboard_bottom[
                                cur_pos[0], last_pos[1] : cur_pos[1]
                            ] = (cur_pos[2] - layer.data.shape[2] / 2)
                            if (
                                widget_thresh.choose_strategy.value
                                == preproc.bottom_strategy_choices[1]
                            ):
                                preproc.napari_portside_bottom[
                                    cur_pos[0], last_pos[1] : cur_pos[1]
                                ] = (layer.data.shape[2] - cur_pos[2])

                last_pos = cur_pos
                preproc.update_bottom_map_napari(cur_pos[0], add_line_width=0)
                bottom_image_layer.data = preproc.bottom_map

                yield
            # on release
            if dragged:
                dragged = False
                # print("mouse release")
            elif (
                0 <= np.round(event.position[1]) < layer.data.shape[1]
                and 0 <= np.round(event.position[2]) < layer.data.shape[2]
            ):
                cur_pos = np.array(np.round(event.position), dtype=int)
                if event.position[2] < layer.data.shape[2] / 2:
                    preproc.napari_portside_bottom[cur_pos[0], cur_pos[1]] = cur_pos[2]
                    if (
                        widget_thresh.choose_strategy.value
                        == preproc.bottom_strategy_choices[1]
                    ):
                        preproc.napari_starboard_bottom[cur_pos[0], cur_pos[1]] = (
                            layer.data.shape[2] / 2 - cur_pos[2]
                        )
                else:
                    preproc.napari_starboard_bottom[cur_pos[0], cur_pos[1]] = (
                        cur_pos[2] - layer.data.shape[2] / 2
                    )
                    if (
                        widget_thresh.choose_strategy.value
                        == preproc.bottom_strategy_choices[1]
                    ):
                        preproc.napari_portside_bottom[cur_pos[0], cur_pos[1]] = (
                            layer.data.shape[2] - cur_pos[2]
                        )
                # print("mouse clicked")
            # set map to trigger drawing
            preproc.update_bottom_map_napari(int(event.position[0]), add_line_width=0)
            bottom_image_layer.data = preproc.bottom_map
            # print("mouse callback ended")

    def target_pick_mouse_callback(layer, event):
        if (
            interaction_modes.mode is not InteractionMode.TARGET_PICK
            or contact_pick_callback is None
            or event.button != 1
        ):
            return
        dragged = False
        yield
        while event.type == "mouse_move":
            dragged = True
            yield
        if not dragged:
            position = event.position
            if hasattr(layer, "world_to_data"):
                position = layer.world_to_data(position)
            result = contact_pick_callback(
                tuple(np.asarray(position).round().astype(int))
            )
            contact = getattr(result, "contact", result)
            draft = getattr(contact, "draft", None)
            anchor = getattr(draft, "anchor", None)
            if anchor is not None:
                marker = display_position_for_anchor(
                    anchor,
                    chunk_size=preproc.chunk_size,
                    display_channel_width=preproc.ping_len,
                )
                contacts_layer.add(np.asarray(marker)[None, :])

    def interaction_dispatcher(layer, event):
        if interaction_modes.mode is InteractionMode.BOTTOM_EDIT:
            return bottom_edit_mouse_callback(layer, event)
        if interaction_modes.mode is InteractionMode.TARGET_PICK:
            return target_pick_mouse_callback(layer, event)
        return None

    # Handle click or drag events separately
    @bottom_image_layer.mouse_drag_callbacks.append
    def click_drag(layer, event):
        return interaction_dispatcher(bottom_image_layer, event)

    # enable the custom callback for all layers
    @sidescan_image_layer.mouse_drag_callbacks.append
    def click_drag(layer, event):
        return interaction_dispatcher(sidescan_image_layer, event)

    @edges_image_layer.mouse_drag_callbacks.append
    def click_drag(layer, event):
        return interaction_dispatcher(edges_image_layer, event)

    @binarized_image_layer.mouse_drag_callbacks.append
    def click_drag(layer, event):
        return interaction_dispatcher(binarized_image_layer, event)

    # run main loop
    # Keep project objects reachable for non-blocking tests and integrations.
    viewer._sidescantools_contact_store = contact_store
    viewer._sidescantools_contact_dock = contact_dock
    viewer._sidescantools_interaction_modes = interaction_modes
    viewer.show(block=block)
    if block:
        contact_store.close()
    return viewer


if __name__ == "__main__":
    chunk_size = 1000
    default_threshold = (
        0.07  # [0.0, 1.0] -> threshold to make sonar img binary for edge detection
    )
    downsampling_factor = 1
    active_dB = False

    filepath = Path("add_path_to_file_here")
    work_dir = "./sidescan_out"
    run_napari_btm_line(
        filepath,
        chunk_size,
        default_threshold,
        downsampling_factor,
        work_dir=work_dir,
        active_dB=active_dB,
    )
