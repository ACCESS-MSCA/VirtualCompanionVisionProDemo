//
//  VirtualCompanionVisionProDemoApp.swift
//  VirtualCompanionVisionProDemo
//
//  Created by harris partaourides on 02/12/2025.
//

import SwiftUI

@main
struct VirtualCompanionVisionProDemoApp: App {

    @State private var appModel = AppModel()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(appModel)
        }

        ImmersiveSpace(id: appModel.immersiveSpaceID) {
            ImmersiveView()
                .environment(appModel)
                .onAppear {
                    appModel.immersiveSpaceState = .open
                }
                .onDisappear {
                    appModel.immersiveSpaceState = .closed
                }
        }
        .immersionStyle(selection: .constant(.full), in: .full)
    }
}
